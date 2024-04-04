---
layout: post
title: DDP(DistributedDataParallel)_2
date: 2024-02-28
tags: [llm]
author: taot
---

## DDP(DistributedDataParallel) 分布式训练2——原理与实践




### 1 分布式编程

一个分布式系统，相对于单机系统，其最大的特征就是，其数据、处理是分布在不同地方的。与此相伴的是，各节点间有交换数据的需求，为此需要定义交换数据的规范、接口。在此基础上，才能构建起分布式计算的大框架。例如google大数据三驾马车之一的`map-reduce`概念，简要地描述，就是将数据分开成N份map到N个地方，并行进行处理；处理完成后，再将结果reduce到一起。

为了满足分布式编程的需求，PyTorch提供了一些分布式基本接口，在torch.distributed中。[torch.distributed代码文档](https://pytorch.org/docs/stable/distributed.html)

#### 1.1 all_reduce

所谓的reduce，就是不同节点各有一份数据，把这些数据汇总到一起。在这里，我们规定各个节点上的这份数据有着相同的shape和data type，并规定汇总的方法是相加。简而言之，就是把各个节点上的一份相同规范的数据相加到一起。

所谓的all_reduce，就是在reduce的基础上，把最终的结果发回到各个节点上。

具体的allreduce实现，要看具体的backend。流行的GPU backend NCCL，all_reduce的实现使用了ring思想。

DDP利用all_reduce，来进行不同进程上的梯度的平均操作。PyTorch提供了几个all_reduce的版本，下面这个就是Ring-Reduce版本：

```python
def all_reduce(tensor,
               op=ReduceOp.SUM,
               group=group.WORLD,
               async_op=False):
    """
    Reduces the tensor data across all machines in such a way that all get
    the final result.
    After the call ``tensor`` is going to be bitwise identical in all processes.
    Arguments:
        tensor (Tensor): Input and output of the collective. The function
            operates in-place.
        op (optional): One of the values from
            ``torch.distributed.ReduceOp``
            enum.  Specifies an operation used for element-wise reductions.
        group (ProcessGroup, optional): The process group to work on
        async_op (bool, optional): Whether this op should be an async op
    Returns:
        Async work handle, if async_op is set to True.
        None, if not async_op or if not part of the group
    """
```


### 2 pytorch 数据结构

#### 2.1 buffer

在PyTorch中，所有的模型都会继承module类。可以说，一个CNN模型，其就是由一系列module嵌套组合而成的。要了解模型，就必须从module下手。下面是module的初始化代码，可以看到，它定义了一系列变量。

```python
# torch.nn.modules.py. line 71. Class module:
    def __init__(self):
        """
        Initializes internal Module state, shared by both nn.Module and ScriptModule.
        """
        torch._C._log_api_usage_once("python.nn_module")

        self.training = True
        self._parameters = OrderedDict()
        self._buffers = OrderedDict()
        self._backward_hooks = OrderedDict()
        self._forward_hooks = OrderedDict()
        self._forward_pre_hooks = OrderedDict()
        self._state_dict_hooks = OrderedDict()
        self._load_state_dict_pre_hooks = OrderedDict()
        self._modules = OrderedDict()
```

module的基本要素可以分为2组，一组是状态，一组是各种各样的hooks。

状态有以下4个东西：
* self.training
  指的是网络是否在训练状态中。
* self._modules
  modules是下属的模块，相当于迭代地定义了self.trainig, self._modules, self._parameters等一系列变量
* self._parameters
  指的是网络的参数
* self._buffers
  不是参数，但也对网络很重要，会被持久化保存的数据。

例如，BatchNorm中的moving mean and variance就是buffer，其优化不是通过梯度反向传播而是通过其他途径。

从本质上讲，当一个模型的网络结构被定义后，其状态就是由parameter和buffer的迭代组合表示的。当我们保存模型，调用model.staic_dict()的时候，我们同时会得到模型的parameter和buffer。

在DDP中，如果我们要在不同进程中维持相同的状态，我们不光要传递parameter的梯度，也要传递buffer。事实上，DDP就是这么做的。当每次网络传播开始前，其都会把master节点上的buffer广播给其他节点，维持状态的统一。

#### 2.2 hook

hook提供了这么一种机制：程序提供hook接口，用户可以写一个hook函数，然后钩在hook接口，即程序的主体上从而可以插入到中间执行。DDP使用hook技术把自己的逻辑插入到module的训练过程中去。

在模型训练时，各个进程通过一种叫Ring-Reduce的方法与其他进程通讯，从而获得所有进程的梯度。PyTorch提供了很多个hook接口，可以用来把 Ring-Reduce 机制插入到 module 中。

parameter在反向梯度计算结束后提供了一个hook接口。DDP把Ring-Reduce的代码写成一个hook函数，插入到这里。每次parameter的反向梯度计算结束后，程序就会调用这个hook函数，从而开启Ring-Reduce流程。因为所有模型都用到parameter，所以DDP模型用hook函数就解决了所有模型的梯度平均问题。

##### 2.2.1 torch.nn.parameter

torch.nn.parameter是torch.Tensor上的一层概念封装，hook机制也是定义在torch.Tensor中的。

##### 2.2.2 torch.tensor.Tensor

DDP的关键代码（即梯度平均）是用C++实现的。但是，在C++、python代码中Tensor都给出了hook接口，实现相似的功能。

Tensor的python hook接口：
```python
# line 200. Class Tensor.
    def register_hook(self, hook):
        r"""Registers a backward hook.
        The hook will be called every time a gradient with respect to the
        Tensor is computed. The hook should have the following signature::
            hook(grad) -> Tensor or None
        The hook should not modify its argument, but it can optionally return
        a new gradient which will be used in place of :attr:`grad`.
        This function returns a handle with a method ``handle.remove()``
        that removes the hook from the module.
        Example::
            >>> v = torch.tensor([0., 0., 0.], requires_grad=True)
            >>> h = v.register_hook(lambda grad: grad * 2)  # double the gradient
            >>> v.backward(torch.tensor([1., 2., 3.]))
            >>> v.grad
             2
             4
             6
            [torch.FloatTensor of size (3,)]
            >>> h.remove()  # removes the hook
        """
```

### 3 DDP 内部实现

#### 3.1 DDP 代码文档链接：

[https://github.com/pytorch/pytorch/blob/v1.5.0/torch/nn/parallel/distributed.py](https://github.com/pytorch/pytorch/blob/v1.5.0/torch/nn/parallel/distributed.py)
[https://github.com/pytorch/pytorch/blob/v1.5.0/torch/csrc/distributed/c10d/reducer.h](https://github.com/pytorch/pytorch/blob/v1.5.0/torch/csrc/distributed/c10d/reducer.h)
[https://github.com/pytorch/pytorch/blob/v1.5.0/torch/distributed/distributed_c10d.py](https://github.com/pytorch/pytorch/blob/v1.5.0/torch/distributed/distributed_c10d.py)
[https://pytorch.org/docs/stable/notes/ddp.html](https://pytorch.org/docs/stable/notes/ddp.html)

#### 3.2 DDP 模式

##### 3.2.1 准备阶段

* 环境准备（就是init_process_group这一步）。各个进程会在这一步，与master节点进行握手，建立连接。
  * 如果连接上的进程数量不足约定的 word_size，进程会一直等待。也就是说，如果你约定了world_size=64，但是只开了6台8卡机器，那么程序会一直暂停在这个地方。
* DDP初始化（也就是model = DDP(model)这一步）
  * 把parameter，buffer从master节点传到其他节点，使所有进程上的状态一致。
  * DDP通过这一步保证所有进程的初始状态一致。所以，请确保在这一步之后，代码不会再修改模型的任何东西，包括添加、修改、删除parameter和buffer
  * 如果每个节点有多卡，则在每张卡上创建模型（类似DP）
* 把parameter进行分组，每一组称为一个bucket。临近的parameter在同一个bucket。
  * 这是为了加速，在梯度通讯时，先计算、得到梯度的bucket会马上进行通讯，不必等到所有梯度计算结束才进行通讯。
* 创建管理器reducer，给每个parameter注册梯度平均的hook。
  * 这一步的具体实现是在C++代码里面的，即reducer.h文件。
* 为可能的SyncBN层做准备

##### 3.2.2 正式训练阶段

在每个step中，DDP模型都会做下面的事情：
* 采样数据，从dataloader得到一个batch的数据，用于当前计算（for data, label in dataloader）。
  * 因为我们的dataloader使用了DistributedSampler，所以各个进程之间的数据是不会重复的。如果要确保DDP性能和单卡性能一致，这边需要保证在数据上，DDP模式下的一个epoch和单卡下的一个epoch是等效的。
* 进行网络的前向计算（prediction = model(data)）
  * 同步各进程状态
    * 对单进程多卡复制模式，要在进程内同步多卡之间的parameter和buffer
  * 同步各进程之间的buffer。
  * 进行真正的前向计算
  * 当DDP参数find_unused_parameter为true时，其会在forward结束时，启动一个回溯，标记出所有没被用到的parameter，提前把这些设定为ready。
    * find_unused_parameter的默认值是false，因为其会拖慢速度。
* 计算梯度（loss.backward()）
  * reducer外面：各个进程各自开始反向地计算梯度。
    * 梯度是反向计算的，所以最后面的参数反而是最先得到梯度的。
  * reducer外面：当某个parameter的梯度计算好了的时候，其之前注册的grad hook就会被触发，在reducer里把这个parameter的状态标记为ready。
  * reducer里面：当某个bucket的所有parameter都是ready状态时，reducer会开始对这个bucket的所有parameter都开始一个异步的all-reduce梯度平均操作。
    * bucket的执行过程也是有顺序的，其顺序与parameter是相反的，即最先注册的parameter的bucket在最后面。
    * 在创建module的时候，务必把先进行计算的parameter注册在前面，后计算的在后面。不然，reducer会卡在某一个bucket等待，使训练时间延长
      * 所谓的参数注册，其实就是创建网络层。也就是要求按照网络计算顺序，依次创建网络层。
  * reducer里面：当所有bucket的梯度平均都结束后，reducer才会把得到的平均grad结果正式写入到parameter.grad里面。
* 优化器optimizer应用gradient，更新参数（optimizer.step()）。
  * 这一步，是和DDP没关系的。

虽然DDP的实现代码与optimizer没有关系，但是关于optimizer有个额外的东西需要说明。更新后的参数最终能在各进程间保持一致，是由以下因素保证的：
* 参数初始值相同
* 参数更新值相同
  * 更新值相同又是由以下因素保证的：
    * optimizer初始状态相同
    * 每次opimizer.step()时的梯度相同。

因为optimizer和DDP是没有关系的，所以optimizer初始状态的同一性是不被DDP保证的！

大多数官方optimizer，其实现在能保证从同样状态的model初始化时，其初始状态是相同的。所以这边我们只要保证在DDP模型创建后才初始化optimizer，就不用做额外的操作。但是，如果自定义optimizer，则需要自己来保证其统一性！

##### 3.2.3 性能没有提升的可能原因

* 检查是否按照单进程单卡的模式启动
* 检查是否使用的是 NCCL 的后端
* 保证不同进程里的模型都是相同结构的；保证parameter（你可以理解为网络层）的创建顺序是一致的。
* 检查是否模型的parameter创建顺序与真实计算顺序一致
* 不允许在产生DDP后，新增、减少、随机修改、替换参数，会造成梯度reduce出错、各进程间的参数不相同、丢失hook机制
* 检查DDP模式下的一个epoch的数据和单卡下的一个epoch的数据是否是等效
* 检查初始状态的同一性：
  * parameter、buffer初始状态同一性
  * optimizer初始状态同一性

##### 3.2.4 DistributedSampler机制

DistributedSampler能够 给不同进程分配数据集的不重叠、不交叉部分。每次epoch我们都会随机shuffle数据集，那么，不同进程之间要怎么保持shuffle后数据集的一致性呢？DistributedSampler的实现方式是，不同进程会使用一个相同的随机数种子，这样shuffle出来的东西就能确保一致。

DistributedSampler使用当前epoch作为随机数种子，从而使得不同epoch下有不同的shuffle结果。所以，每次epoch开始前都要调用一下sampler的set_epoch方法，这样才能让数据集随机shuffle起来。

DistributedSampler的核心源代码：
```python
# line 56
    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        if self.shuffle:
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))


        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)
# line 79
    def set_epoch(self, epoch):
        self.epoch = epoch
```
