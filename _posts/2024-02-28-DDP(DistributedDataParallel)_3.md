---
layout: post
title: DDP(DistributedDataParallel)_3
date: 2024-02-29
tags: [LLM]
author: taot
---

## DDP(DistributedDataParallel) 分布式训练3——实践与技巧


### 1 在 DDP 中引入 syncBN

#### 1.1 syncBN

SyncBN就是Batch Normalization(BN)。其跟一般所说的普通BN的不同在于工程实现方式：SyncBN能够完美支持多卡训练，而普通BN在多卡模式下实际上就是单卡模式。

BN中有moving mean和moving variance这两个buffer，这两个buffer的更新依赖于当前训练轮次的batch数据的计算结果。但是在普通多卡DP模式下，各个模型只能拿到自己的那部分计算结果，所以在DP模式下的普通BN被设计为只利用主卡上的计算结果来计算moving mean和moving variance，之后再广播给其他卡。这样，实际上BN的batch size就只是主卡上的batch size那么大。当模型很大、batch size很小时，这样的BN无疑会限制模型的性能。

为了解决这个问题，PyTorch新引入了一个叫SyncBN的结构，利用DDP的分布式计算接口来实现真正的多卡BN。


#### 1.1.1 syncBN 原理

SyncBN利用分布式通讯接口在各卡间进行通讯，从而能利用所有数据进行BN计算。为了尽可能地减少跨卡传输量，SyncBN做了一个关键的优化，即只传输各自进程的各自的 小batch mean和 小batch variance，而不是所有数据。

具体流程请见下面：
* 前向传播
  * 在各进程上计算各自的 小batch mean和小batch variance
  * 各自的进程对各自的 小batch mean和小batch variance进行all_gather操作，每个进程都得到全局量。
    * 只传递mean和variance，而不是整体数据，可以大大减少通讯量，提高速度。
  * 每个进程分别计算总体mean和总体variance，得到一样的结果
    * 在数学上是可行的
  * 接下来，延续正常的BN计算。
    * 因为从前向传播的计算数据中得到的batch mean和batch variance在各卡间保持一致，所以，running_mean和running_variance就能保持一致，不需要显式地同步了！
* 后向传播：和正常的一样


#### 1.1.2 SyncBN与DDP的关系

当前PyTorch SyncBN只在DDP单进程单卡模式中支持。SyncBN用到 all_gather这个分布式计算接口，而使用这个接口需要先初始化DDP环境。

三个需要注意的点：
* 为SyncBN层做准备，实际上就是检测当前是否是DDP单进程单卡模式，如果不是，会直接停止。
* SyncBN需要在DDP环境初始化后初始化，但是要在DDP模型前就准备好。
* 当前PyTorch SyncBN只支持DDP单进程单卡模式的原因：从SyncBN原理中我们可以看到，其强依赖了all_gather计算，而这个分布式接口当前是不支持单进程多卡或者DP模式的。

#### 1.1.3 syncBN 使用

```python
# DDP init
dist.init_process_group(backend='nccl')

# 按照原来的方式定义模型，这里的BN都使用普通BN就行了。
model = MyModel()
# 引入SyncBN，这句代码，会将普通BN替换成SyncBN。
model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)

# 构造DDP模型
model = DDP(model, device_ids=[local_rank], output_device=local_rank)
```

##### 1.1.3.1 convert_sync_batchnorm的原理：

torch.nn.SyncBatchNorm.convert_sync_batchnorm会搜索model里面的每一个module，如果发现这个module是、或者继承了torch.nn.modules.batchnorm._BatchNorm类，就把它替换成SyncBN。也就是说，如果你的Normalization层是自己定义的特殊类，没有继承过_BatchNorm类，那么convert_sync_batchnorm是不支持的，需要你自己实现一个新的SyncBN

convert_sync_batchnorm的源码:

```python
 @classmethod
    def convert_sync_batchnorm(cls, module, process_group=None):
        r"""Helper function to convert all :attr:`BatchNorm*D` layers in the model to
        :class:`torch.nn.SyncBatchNorm` layers.
        """
        module_output = module
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module_output = torch.nn.SyncBatchNorm(module.num_features,
                                                   module.eps, module.momentum,
                                                   module.affine,
                                                   module.track_running_stats,
                                                   process_group)
            if module.affine:
                with torch.no_grad():
                    module_output.weight = module.weight
                    module_output.bias = module.bias
            module_output.running_mean = module.running_mean
            module_output.running_var = module.running_var
            module_output.num_batches_tracked = module.num_batches_tracked
        for name, child in module.named_children():
            module_output.add_module(name, cls.convert_sync_batchnorm(child, process_group))
        del module
        return module_output
```

convert的过程中，新的SyncBN复制了原来的BN层的所有参数。


### 2 DDP下的Gradient Accumulation的进一步加速

#### 2.1 Gradient Accmulation

Gradient Accumulation，即梯度累加，是一种增大训练时batch size的技术。

##### 2.2.1 Gradient Accumulation 加速原理

DDP下的gradient accumulation：
```python
# 单卡模式，即普通情况下的梯度累加
for 每次梯度累加循环
    optimizer.zero_grad()
    for _ in range(K):
        prediction = model(data)
        loss = loss_fn(prediction, label) / K  # 除以K，模仿loss function中的batchSize方向上的梯度平均，如果本身就没有的话则不需要。
        loss.backward()  # 积累梯度，不应用梯度改变
    optimizer.step()  # 应用梯度改变
```

DDP的gradient all_reduce阶段发生在loss_fn(prediction, label).backward()。这意味着，在梯度累加的情况下，假设一次梯度累加循环有K个step，每次梯度累加循环会进行K次 all_reduce！

但事实上，每次梯度累加循环只会有一次 optimizer.step()，即只应用一次参数修改，这意味着在每一次梯度累加循环中，我们其实只要进行一次gradient all_reduce即可满足要求，有K-1次 all_reduce被浪费了！而每次 all_reduce的时间成本是很高的！

**解决问题的思路在于，对前K-1次step取消其梯度同步。**DDP给我们提供了一个暂时取消梯度同步的context函数 no_sync()，[源码](https://github.com/pytorch/pytorch/blob/main/torch/nn/parallel/distributed.py#L548)。在这个context下，DDP不会进行梯度同步。所以，我们可以这样实现加速：

```python
model = DDP(model)

for 每次梯度累加循环
    optimizer.zero_grad()
    # 前accumulation_step-1个step，不进行梯度同步，累积梯度。
    for _ in range(K-1)::
        with model.no_sync():
            prediction = model(data)
            loss = loss_fn(prediction, label) / K
            loss.backward()  # 积累梯度，不应用梯度改变
    # 第K个step，进行梯度同步
    prediction = model(data)
    loss = loss_fn(prediction, label) / K
    loss.backward()  # 积累梯度，不应用梯度改变
    optimizer.step()
```

同时兼容单卡、DDP模式的写法：
```python
from contextlib import nullcontext
# 如果你的python版本小于3.7，请注释掉上面一行，使用下面这个：
# from contextlib import suppress as nullcontext

if local_rank != -1:
    model = DDP(model)

optimizer.zero_grad()
for i, (data, label) in enumerate(dataloader):
    # 只在DDP模式下，轮数不是K整数倍的时候使用no_sync
    my_context = model.no_sync if local_rank != -1 and i % K != 0 else nullcontext
    with my_context():
        prediction = model(data)
        loss = loss_fn(prediction, label) / K
        loss.backward()  # 积累梯度，不应用梯度改变
    if i % K == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### 3 多机多卡环境下的inference加速

在训练中每几个epoch我们会跑一下inference、测试一下模型性能。在DDP多卡训练环境下，能不能利用多卡来加速inference速度呢？有一堆数据要跑一些网络推理，拿到inference结果。DP下多卡加速比太低，能不能利用DDP多卡来加速呢？

#### 3.1 多机多卡环境下的inference加速方法

测试和训练的不同在于：
* 测试的时候不需要进行梯度反向传播，inference过程中各进程之间不需要通讯。
* 测试的时候，不同模型的inference结果、性能指标的类型多种多样，没有统一的形式。
  * 很难定义一个统一的框架，像训练时model=DDP(model)那样方便地应用DDP多卡加速。

可以在各个进程中各自进行单卡的inference，然后把结果收集到一起。单卡inference很简单，我们甚至可以直接用DDP包装前的模型。问题其实只有两个：
* 我们要如何把数据split到各个进程中
* 我们要如何把结果合并到一起

#### 3.2 把数据split到各个进程中：新的data sampler

在训练的时候，我们用的 torch.utils.data.distributed.DistributedSampler帮助我们把数据不重复地分到各个进程上去。但是，其分的方法是：每段连续的N个数据，拆成一个一个，分给N个进程，所以每个进程拿到的数据不是连续的。这样，不利于我们在inference结束的时候将结果合并到一起。

因此，这里我们需要实现一个新的data sampler。它的功能，是能够连续地划分数据块，不重复地分到各个进程上去。

```python
# 来源：https://github.com/huggingface/transformers/blob/447808c85f0e6d6b0aeeb07214942bf1e578f9d2/src/transformers/trainer_pt_utils.py
class SequentialDistributedSampler(torch.utils.data.sampler.Sampler):
    """
    Distributed Sampler that subsamples indicies sequentially,
    making it easier to collate all results at the end.
    Even though we only use this sampler for eval and predict (no training),
    which means that the model params won't have to be synced (i.e. will not hang
    for synchronization even if varied number of forward passes), we still add extra
    samples to the sampler to make it evenly divisible (like in `DistributedSampler`)
    to make it easy to `gather` or `reduce` resulting tensors at the end of the loop.
    """

    def __init__(self, dataset, batch_size, rank=None, num_replicas=None):
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.batch_size = batch_size
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.batch_size / self.num_replicas)) * self.batch_size
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        # add extra samples to make it evenly divisible
        indices += [indices[-1]] * (self.total_size - len(indices))
        # subsample
        indices = indices[self.rank * self.num_samples : (self.rank + 1) * self.num_samples]
        return iter(indices)

    def __len__(self):
        return self.num_samples
```

#### 3.3 把结果合并到一起: all_gather

通过torch.distributed提供的分布式接口all_gather，我们可以把各个进程的prediction结果集中到一起。

存在着千奇百怪的神经网络模型，有着千奇百怪的输出，所以，把数据集中到一起不是一件容易的事情。但是，如果你的网络输出在不同的进程中有着一样的大小，那么这个问题就好解多了。下面给一个方法，其要求网络的prediction结果在各个进程中的大小是一模一样的：

```python
# 合并结果的函数
# 1. all_gather，将各个进程中的同一份数据合并到一起。
#   和all_reduce不同的是，all_reduce是平均，而这里是合并。
# 2. 要注意的是，函数的最后会裁剪掉后面额外长度的部分，这是之前的SequentialDistributedSampler添加的。
# 3. 这个函数要求，输入tensor在各个进程中的大小是一模一样的。
def distributed_concat(tensor, num_total_examples):
    output_tensors = [tensor.clone() for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(output_tensors, tensor)
    concat = torch.cat(output_tensors, dim=0)
    # truncate the dummy elements added by SequentialDistributedSampler
    return concat[:num_total_examples]
    
```

完整流程：
```python
## 构造测试集
# 假定我们的数据集是这个
transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
my_testset = torchvision.datasets.CIFAR10(root='./data', train=False, 
        download=True, transform=transform)
# 使用我们的新sampler
test_sampler = SequentialDistributedSampler(my_testset, batch_size=16)
testloader = torch.utils.data.DataLoader(my_testset, batch_size=16, sampler=test_sampler)

# DDP和模型初始化，略。
# ......

# 正式训练和evaluation
for epoch in range(total_epoch_size):
    # 训练代码，略
    # .......
    # 开始测试
    with torch.no_grad():
        # 1. 得到本进程的prediction
        predictions = []
        labels = []
        for data, label in testloader:
            data, label = data.to(local_rank), label.to(local_rank)
            predictions.append(model(data))
            labels.append(label)
        # 进行gather
        predictions = distributed_concat(torch.concat(predictions, dim=0), 
                                         len(test_sampler.dataset))
        labels = distributed_concat(torch.concat(labels, dim=0), 
                                    len(test_sampler.dataset))
        # 3. 现在我们已经拿到所有数据的predictioin结果，进行evaluate！
        my_evaluate_func(predictions, labels)
```

* 如果我们的目的只是得到性能数字，那么，我们甚至可以直接在各个进程中计算各自的性能数字，然后再合并到一起。
* 可以单向地把predictions、labels集中到 rank=0的进程，只在其进行evaluation并输出。PyTorch也提供了相应的接口（[send和recv](https://pytorch.org/docs/stable/distributed.html)）。


### 4 保证DDP性能：确保数据的一致性

#### 4.1 性能期望

从原理上讲，当没有开启SyncBN时，（或者更严格地讲，没有BN层；但一般有的话影响也不大），以下两种方法训练出来的模型应该是性能相似的：
* 进程数为N的DDP训练
* accumulation为N、其他配置完全相同的单卡训练

如果我们发现性能对不上，那么，往往是DDP中的某些设置出了问题。

DDP训练时，数据的一致性必须被保证：各个进程拿到的数据，要像是accumulation为N、其他配置完全相同的单卡训练中同个accumulation循环中不同iteration拿到的数据。想象一下，如果各个进程拿到的数据是一样的，或者分布上有任何相似的地方，那么，这就会造成训练数据质量的下降，最终导致模型性能下降。

#### 4.2 需要注意的点

为保证实验的可复现性，一般我们会在代码在开头声明一个固定的随机数种子，从而使得同一个配置下的实验，无论启动多少次，都会拿到同样的结果。

```python
import random
import numpy as np
import torch

def init_seeds(seed=0, cuda_deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    if cuda_deterministic:  # slower, more reproducible
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:  # faster, less reproducible
        cudnn.deterministic = False
        cudnn.benchmark = True
        

def main():
    # 一般都直接用0作为固定的随机数种子。
    init_seeds(0)
```

在DDP训练中，如果还是像以前一样，使用0作为随机数种子，不做修改，就会造成以下后果：
* DDP的N个进程都使用同一个随机数种子
* 在生成数据时，如果我们使用了一些随机过程的数据扩充方法，那么，各个进程生成的数据会带有一定的同态性。
  * 例如，YOLOv5会使用mosaic数据增强（从数据集中随机采样3张图像与当前的拼在一起，组成一张里面有4张小图的大图）。这样，因为各卡使用了相同的随机数种子，你会发现，各卡生成的图像中，除了原本的那张小图，其他三张小图都是一模一样的！
* 同态性的数据，降低了训练数据的质量，也就降低了训练效率！最终得到的模型性能，很有可能是比原来更低的。

因此，需要给不同的进程分配不同的、固定的随机数种子：

```python
def main():
    rank = torch.distributed.get_rank()
    init_seeds(1 + rank)
```


### 5 和DDP有关的小技巧

#### 5.1 控制不同进程的执行顺序

一般情况下，各个进程是各自执行的，速度有快有慢，只有在gradient all-reduce的时候，快的进程才会等一下慢的进程，也就是进行同步。那么，如果我们需要在其他地方进行同步呢？

比如说，在加载数据前，如果数据集不存在，我们要下载数据集：
* 我们只需要在唯一一个进程中开启一次下载
* 我们需要让其他进程等待其下载完成，再去加载数据

torch.distributed提供了一个barrier()的接口，利用它我们可以同步各个DDP中的各个进程！当使用barrier函数时，DDP进程会在函数的位置进行等待，知道所有的进程都跑到了 barrier函数的位置，它们才会再次向下执行。

* 只在某进程执行，无须同步

```python
if rank == 0:
    code_only_run_in_rank_0()
```

* 简单的同步

```python
code_before()
# 在这一步同步
torch.distributed.barrier()
code_after()
```

* 在某个进程中执行A操作，其他进程等待其执行完成后再执行B操作

```python
if rank == 0:
    do_A()
    torch.distributed.barrier()
else:
    torch.distributed.barrier()
    do_B()
```

* 在某个进程中优先执行A操作，其他进程等待其执行完成后再执行A操作

```python
from contextlib import contextmanager

@contextmanager
def torch_distributed_zero_first(rank: int):
    """Decorator to make all processes in distributed training wait for each local_master to do something.
    """
    if rank not in [-1, 0]:
        torch.distributed.barrier()
    # 这里的用法其实就是协程的一种
    yield
    if rank == 0:
        torch.distributed.barrier()

    # 之后接可以使用上面的包装
    with torch_distributed_zero_first(rank):
    if not check_if_dataset_exist():
        download_dataset()
    load_dataset()
```

#### 5.2 避免DDP带来的冗余输出

当我们在自己的模型中加入DDP模型时，第一的直观感受肯定是，终端里的输出变成了N倍了。这是因为我们现在有N个进程在同时跑整个程序。因为各个进程的速度并不一样快，在茫茫的输出海洋中，我们难以debug、把控实验状态。

一种解决方法：
logging模块+输出信息等级控制。即用logging输出代替所有print输出，并给不同进程设置不同的输出等级，只在0号进程保留低等级输出。

```python
import logging

# 给主要进程（rank=0）设置低输出等级，给其他进程设置高输出等级。
logging.basicConfig(level=logging.INFO if rank in [-1, 0] else logging.WARN)
# 普通log，只会打印一次。
logging.info("This is an ordinary log.")
# 危险的warning、error，无论在哪个进程，都会被打印出来，从而方便debug。
logging.error("This is a fatal log!")
```