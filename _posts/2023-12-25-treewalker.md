---
layout: post
title: treewalker
date: 2023-12-25
tags: [compiler]
author: taot
---

## TreeWalker加速推理调度算法

TreeWalker 是基于自研 NPU，结合软硬件架构，设计的一套加速网络推理速度、减小内存消耗的调度算法。

相比于传统的 AI 编译器，我们使用整体层融合的方式，能够加速整网的推理速度，还能够减少内存的消耗。我们的具体做法是这样的：对于输入的 feature map，我们会对其进行切块处理。我们根据输出来反推输入，保证每一层的输出数据都能够满足下一层计算所需。这样，通过每一层只计算最少的数据量，从而尽快走到网络的最后一层。使用这样的整体层融合的方法，并对在具体实现中进行相关的优化，能够提高 50% 以上的推理速度，大型网络的内存消耗峰值低于 6MB。为了加快推理速度，减少内存消耗，主要进行了这样的一些优化：

*   interval timer，减少相邻算子并行等待时间
    *   如果是没有数据依赖的算子，可以并行
    *   如果有数据依赖，那么需要考虑 interval time。前一层被调度到，就会被置为 ready，这个信息会被传递给 sspu。前一层的输出和当前层的输入，有数据依赖，需要考虑最大 delay。即起前一层地写完成到当前层的读完成，这之间的时间差。一个简单的方式：前一层输入快一拍。这样会导致 memory 消耗更大，但是更容易统计。
*   采用非模拟的方式，来加快推理。每层的分块分为 first，middle 和 last 三类。middle 代表当前层能够保持当前的状态稳定执行。只需要模拟每一层的 first 块和一个 middle 块即可，就可以根据 first 和 middle 的尺寸，计算出相关的控制参数。从而减少推理耗时。这样会有 tile 计算流产生方面的问题。
*   根据实际情况进行局部的层融合，减少内存消耗。针对 gap 的情况，通过调度算法控制，可以使数据生产慢的一端多走几次或者是每次计算的数据块更大，使得两个分支的数据供给速度平衡，从而消除 gap 带来的额外内存消耗。
*   也可以通过增大 tile 的尺寸来加快推理，但是会增大内存消耗。
*   通过算子顺序重排，优化分支结构的数据量，从而减小内存消耗。
*   改进想法：因为分为 first，middle，last 三类 tile，控制参数复杂，并且走出来的 tile 计算流不规整，不利于 计算流指令的编码压缩。可以只有 middle 和 last，从而使 TreeWalker 走的更加规整，更有利于 tile isa 的编码压缩

1、pre_tree_walker workflow

*   根据 feature map  的输出是尺寸，先来计算 middle  tile 的输出参数，在计算出相应的 first 参数和 last 参数。
    *   从 TILE_H_MAX 开始尝试，不满足两个条件则减重新尝试：tile_oh < 16; tile_ih > tile_h_limiit
*   middle 参数的计算是先设定最大值，再根据前后层的数据依赖的数据一致性要求，若满足则就是当前的值，否则就需要多参数进行减小处理，直到符合数据依赖以及数据一致性要求
*   first 的参数一般设置为和 middle 一致，如果有从编译的时候传递的参数，那么需要使用设置的参数来替换；
    *   DMA_IN  层，根据前一层的 moh  和 DMA_IN 层总高，推导出 fih
    *   其它层：根据 DMA_IN 层的 foh 和 moh 来根据数据依赖来直接计算出其它层的 foh
*   last 参数的计算是根据 feature map 减掉 first 和所有的 middle 之后剩下的值

2、treewalker work flow

*   使用 pre_tree_walker 计算的 first 值
*   根据编译器能够支持的最大的 middle 值，开始模拟硬件的 TreeWalker 行为。模拟硬件 TreeWalker 行为过程中，会统计 OCM 消耗，也会根据所有层的数据依赖和数据一致性要求，来检查现在的 middle 参数是否合理；不合理则回去参数减小，继续模拟，直至满足为止。在这个过程中，也会考虑不同的 OP 的一些特殊的限制
*   根据模拟硬件 TreeWalker 的行为计算出来的 middle、last，以及直接使用的 pre_tree_walker 的 first 参数，对 cmem 中的 layer_args 计算需要的一套控制参数进行计算
*   在模拟硬件 TreeWalker 的过程中，还会得出 tile sequence，即计算流顺序
*   根据 TreeWalker 过程中的参数计算结果，可以分析出各种 buffer 的需求量，进而对 OCM 进行分配

3、TreeWalker tile mode 的特点

*   每层的计算量可调，提高 MAC 利用率
*   更快走到最后一层
*   降低内存消耗
*   参数控制复杂
*   内存管理复杂

4、TreeWalker

*   模拟硬件 TreeWalker 的 tile sequence
*   产生相关的控制参数，后续用于产生 layer command 和 TreeWalker 相关的参数配置
*   Tensor buffer 管理：
    *   数据依赖的检查
    *   内存占用峰值统计
*   栈管理机制

TreeWalker 对网络推理的加速和 OCM 消耗的优化：

### 加速网络推理

整体的层融合、interval timer、局部层融合、非模拟 TreeWalker、增大 block 的 shape



### 减少 OCM 消耗

整体的层融合 + block 切分、block 内部可以选择优先方向，使用寄存器、small GAP 情况算子重排、small  gap 情况，控制近端和远端的数据消耗平衡



### ZEKU Z3 & Z4 compiler 技术点总结

1、line partition / branch partition / branchx partition

*   line: 一个一个 merge，直到放不下
*   branch: 一个一个 merge，直到放不下，可以包含分支
*   branchx: 二分切，直到放得下
*   按照这样的方式切子图之后，如何保证性能最优：目前是在 ocm 用满的情况下就认为是最优，我们的 nne lib 只返回 TRUE 或者 FALSE，没有 cycle 计算。实际上会有 performance model 来进行评估，但是我们的 performance model 还没有添加进来。Z3 的所有算子的 cycle 都是 1，只考虑了内存。

2、VSP / mmp

*   mmp：存放 weight
*   vsp：weight 以外的

3、compiler workflow

protobuf 产生jobdef；frontend 解析模型，对模型进行一系列的 pass 处理，如子图拆分，IR转换等操作；前端解析到的信息填充进 jobdef 里面；经过所有的优化阶段的 pass 之后，会进行 lowering，生成包含 qid，opcode，控制参数等的 dump  的 cmem_structrue，生成给硬件和 model 使用的信息，loadable

### 动态 shape 支持方案

### tile ISA 压缩算法

