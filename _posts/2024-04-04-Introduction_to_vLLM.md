---
layout: post
title: Introduction_to_vLLM
date: 2024-04-04
tags: [llm]
author: taot
---


## vLLM介绍

vLLM是伯克利大学LMSYS组织开源的大语言模型高速推理框架，旨在极大地提升实时场景下的语言模型服务的吞吐与内存使用效率。vLLM是一个快速且易于使用的库，用于 LLM 推理和服务，可以和HuggingFace 无缝集成。vLLM利用了全新的注意力算法「PagedAttention」，有效地管理注意力键和值。

vLLM 的特点和优势：
* 采用了 PagedAttention，可以有效管理 attention 的 keys、values
* 吞吐量最多可以达到 huggingface 实现的24倍，文本生成推理（TGI）高出3.5倍，并且不需要对模型结构进行任何的改变

### 1 PagedAttention

#### 1.1 背景

* LLM 的推理，最大的瓶颈在于显存。
* 自回归模型的 keys 和 values 通常被称为 KV cache，这些 tensors 会存在 GPU 的显存中，用于生成下一个 token。
* 这些 KV cache 都很大，并且大小是动态变化的，难以预测。已有的系统中，由于显存碎片和过度预留，浪费了60%-80%的显存。

#### 1.2 实现 受到操作系统中，虚拟内存和分页经典思想的启发

* PagedAttention 允许在不连续的内存空间中存储连续的 keys 和 values。 具体来说，PagedAttention 会将每个序列的 KV cache 划分为块，每个块包含固定数量 tokens 的 keys 和 values。 在注意力计算过程中，PagedAttention 内核有效地识别并获取这些块。
* 分块之后，这些 KV cache 不再需要连续的内存，从而可以像在操作系统的虚拟内存中一样，更灵活地对这些 KV cache 进行管理。
* PagedAttention 对于显存的利用接近理论上的最优值（浪费比例低于4%）。通过对显存进行更好的管理，可以使得单次可以使用更大的 batch size，从而进一步利用 GPU 的并行计算能力。


### 2 vLLM 离线推理流程
vLLM 整体框架： 
![在这里插入图片描述](https://img-blog.csdnimg.cn/direct/32e432ad53c94a558bafae21cc6dd994.png#pic_center)
其中的关键技术点包括：
* KVCache 显存优化
* PagedAttention
* Continuous Batching

#### 2.1 LLM 和 LLM Engine

LLM 是对 LLM serving 部分的封装，也是核心部分。首先它会初始化这个类。初始化过程中大部分参数都会被用来构造 EngineArgs，这是一个 dataclass，封装了 Engine 的初始化参数。然后构建 LLM Engine。一个 LLM 只有一个 LLM Engine，所以它就是对 Engine 再包一层。不过按照作者的这个设计意思，LLM Engine 也可以单提出来使用。

初始化 LLM Engine 时候会先调用 create_engine_configs 将 EngineArgs 分解成 ModelConfig，CacheConfig， ParallelConfig 和 SchedulerConfig。其中：

* ModelConfig 包括了对 model 和 tokenizer 的定义，dtype 和随机数 seed 以及是否用 pretrained weights 还是 dummy weights 等。
* CacheConfig 包括 block_size（每个 block 多大）， gpu_utilization（GPU 利用率，后面 allocate 的时候占多少 GPU）和 swap_space（swap 的空间大小）。默认 block_size=16，swap_space=4GiB。
* ParallelConfig 包括了 tensor_parallel_size 和 pipeline_parallel_size，即张量并行和流水线并行的 size，由于我们是单卡，这两个都是 1。
* SchdulerConfig 包括了 max_num_batched_tokens（一个 iteration 最多处理多少个 tokens），max_num_seqs（一个 iteration 最多能处理多少数量的 sequences）以及 max_seq_len（最大生成多长的 context length，也就是一个 sequence 的最长长度，包含 prompt 部分和 generated 部分）。

然后对于每个 device（也即每张卡 / 每个 rank）创建一个 Worker。Worker 是运行 model 的单位。一个 Engine 管理所有的 workers。同时给这个 engine 创建它的 scheduler，以及初始化这个 engine 的 KV cache。

#### 2.2 workers

Worker 是对单个 GPU 的抽象。

Engine 通过调用 _run_workers("<method_name>", *args, get_all_outputs, **kwargs) 来在 所有 workers 上执行方法。如果 get_all_outputs 设成 True，那么它会将所有 workers 的返回结果包装成 List 来返回。否则，它只会返回第一个 worker 的结果，并且 assert 所有 workers 的输出都是一样的。在实际执行中主要会调用如下方法（方法名, get_all_outputs=False/True）：

* profile_num_avaiable_block，True：通过一次 “试运行” 来 profile peak memory。 每张卡的 blocks 个数可能不同（显存不同），所以需要 get all outputs。由于 vLLM 使用一个中心化的管理单元，因此我们会对 profile 出来的 blocks 个数取 min。
* init_cache_engine，False：初始化 cache engine。由于返回 None，所以不需要 get all outputs。
* execute_model ，False：执行模型。这里虽然是分布式 inference，但是最后 output 都会被 reduce，所以 get all outputs 也设成 False 就好了。

Worker 初始化阶段会初始化模型和一些 distributed 相关的东西。


#### 2.3 Cache Engine

用于管理 KV Cache 的单元。

初始化时候，它先根据之前 profile 的数据（cpu/gpu blocks数）来 allocate cache。然后再给 caching 操作初始化一个 CUDA Stream，以及给每一个 layer 初始化一个 cuda event 来用做 stream synchronization。

在 vLLM 里，每个 key block 的 shape 是 [num_heads, head_size // x, block_size, x]，其中 x 是 16 // dtype 的大小。也就是说 fp32 时 x=4，fp16 时 x=8。每个 value block 的 shape 是 [num_heads, head_size, block_size]。

在分配 cpu cache 时候，默认是会用 pin memory 的（除非在 WSL）。


cache engine 里支持了其它两个操作：

* copy。由专门的 cu 函数 copy_blocks 支持。
* swap_in 和 swap_out。有点像操作系统里的 swap 概念。in 就是 cpu to gpu，out 就是 gpu to cpu。内部实现由专门的 cu 函数 swap_blocks 支持。


相关的 cu 函数，实现在 csrc/cache_kernels.cu 中:

* swap_blocks(src, dst, block_mapping)： for 一遍 block_mapping，对于每个 [src, dst] pair（block number to block number）做一个单 block 的 copy。支持 GPU to GPU（必须是同一个 GPU 上），GPU to CPU，CPU to GPU。
* copy_blocks(key_caches, value_caches, block_mapping)：这里的 mapping 是 int->list[int]，也就是一个 src block idx 对应多个 dst block idx。copy 的过程用一个 global kernel 并行实现。
* reshape_and_cache(key, value, key_cache, value_cache, slot_mapping)
* gather_cached_kv(key, value, key_cache, value_cache, slot_mapping)
#### 2.4 memory sharing

* memory sharing 是 PagedAttention 的另一个关键特性。
* 当用单个 prompt 产出多个不同的序列时，可以共享计算量和显存。
* 通过将不同序列的 logical blocks 映射到同一个 physical blocks，可以实现显存共享。
* 为了保证共享的安全性，对于 physical blocks 的引用次数进行统计，并实现了 Copy-on-Write 机制。
* 这种内存共享机制，可以大幅降低复杂采样算法对于显存的需求（最高可下降55%），从而可以提升2.2倍的吞吐量。


