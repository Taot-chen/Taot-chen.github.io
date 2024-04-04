---
layout: post
title: PyTorch-multi-card_training_related_concepts
date: 2024-02-29
tags: [pytorch]
author: taot
---

## pytorch 多卡训练相关概念


### 1、World，Rank，Local Rank

#### 1.1 world

World可以认为是一个集合，由一组能够互相发消息的进程组成。
world size就表示这组能够互相通信的进程的总数。


#### 1.2 rank

Rank可以认为是这组能够互相通信的进程在World中的序号。

#### 1.3 local rank

Local Rank可以认为是这组能够互相通信的进程在它们相应主机(Host)中的序号。

不准确但较容易理解的：
* world，所有能够通信的 gpu 集合。一般多卡所有卡都是可以通信的，因此可以简单理解为所有 GPU 的集合
* world size，可以简单理解为所有 GPU 的数目
* rank，gpu 在所有 gpu 中的序号
* local rank，gpu 单所在主机中的 gpu 里面的序号

### 2 模型并行与数据并行

* 模型并行（model parellelism）：如果模型特别大，GPU显存不够，无法将一个显存放在GPU上，需要把网络的不同模块放在不同GPU上，这样可以训练比较大的网络。
* 数据并行（data parellelism）：将整个模型放在一块GPU里，再复制到每一块GPU上，同时进行正向传播和反向误差传播。相当于加大了batch_size。

### 3 BN 在不同设备之间的同步

假设batch_size=2，每个GPU计算的均值和方差都针对这两个样本而言的。而BN的特性是：batch_size越大，均值和方差越接近与整个数据集的均值和方差，效果越好。

使用多块GPU时，会计算每个BN层在所有设备上输入的均值和方差。如果GPU1和GPU2都分别得到两个特征层，那么两块GPU一共计算4个特征层的均值和方差，可以认为batch_size=4。注意：如果不用同步BN，而是每个设备计算自己的批次数据的均值方差，效果与单GPU一致，仅仅能提升训练速度；如果使用同步BN，效果会有一定提升，但是会损失一部分并行速度。

### 4 混合精度

在混合精度训练上，Apex 的封装十分优雅。直接使用 amp.initialize 包装模型和优化器，apex 就会自动帮助我们管理模型参数和优化器的精度了，根据精度需求不同可以传入其他配置参数。

```python
from apex import amp

model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
```

其中 opt_level 为精度的优化设置:

* O0：纯FP32训练，可以作为accuracy的baseline；
* O1：混合精度训练（推荐使用），根据黑白名单自动决定使用FP16（GEMM, 卷积）还是FP32（Softmax）进行计算。
* O2：“几乎FP16”混合精度训练，不存在黑白名单，除了Batch norm，几乎都是用FP16计算。
* O3：纯FP16训练，很不稳定，但是可以作为speed的baseline；

### 4 记录Loss曲线

在我们使用多进程时，每个进程有自己计算得到的Loss，我们在进行数据记录时，希望对不同进程上的Loss取平均（也就是 map-reduce 的做法），对于其他需要记录的数据也都是一样的做法：

```python
def reduce_tensor(tensor: torch.Tensor) -&gt; torch.Tensor:
    rt = tensor.clone()
    distributed.all_reduce(rt, op=distributed.reduce_op.SUM)
    rt /= distributed.get_world_size()
    return rt

# calculate loss
loss = criterion(predict, labels)
reduced_loss = reduce_tensor(loss.data)
train_epoch_loss += reduced_loss.item()

注意在写入TensorBoard的时候只让一个进程写入就够了：

# TensorBoard
if args.local_rank == 0:
    writer.add_scalars('Loss/training', {
        'train_loss': train_epoch_loss,
        'val_loss': val_epoch_loss
    }, epoch + 1)
```

