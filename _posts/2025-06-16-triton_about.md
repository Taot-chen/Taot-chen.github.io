---
layout: post
title: triton_about
date: 2025-06-16
tags: [triton]
author: taot
---


### 1 Triton 算子代码组成

一个 Triton 算子的实现代码通常分为两个部分：

* **计算准备阶段**：涉及输入张量的预处理（如转换为连续布局张量），计算输出张量的形状并分配内存，以及设置运行参数（如 grid 和 BLOCK_SIZE）。
* **核函数调用**：在 GPU 上实现计算逻辑。


#### 1.1 计算任务划分

Triton 核函数的编程模式是面向 CTA（也即 Thread Block，线程块）的编程。线程及更低层次的硬件细节被 Triton 隐藏，可以只专注于算法设计和线程块的划分。


线程块的划分，本质是对计算资源的规划，它与数据张量的切分方案密不可分。对于 point wise 算子，输入张量与输出张量形状相同，输入张量中每个元素通过相同的运算规则计算出输出张量的对应元素，此时线程块的划分就是数据张量的划分；对于较复杂一些的算子，一个线程块可能处理多个数据块。

在硬件和算子算法确定的情况下，线程块大小的划分是影响 GPU 上执行时间的关键因素。划分方案的效率高低，受到多种因素的影响，包括但不限于数据张量的布局、缓存的有效利用、算法的计算顺序以及显卡硬件的具体限制等。


### 2 Triton Optimizer

Triton 主要有三个部分的优化：

* Triton IR的优化；
* Triton IR到TritonGPU IR的转换；
* TritonGPU IR的优化；

其中 TritonGPU IR 是与硬件GPU相关的IR，该IR包含了与GPU 相关的信息，如硬件相关的Op以及layout信息。


#### 2.1 Triton IR 优化 pass

Triton IR 上的优化主要是关于计算本身的，与硬件无关的优化，主要包含如下 Pass:

* `Inliner Pass`：将 Kernel 内调用的子函数 Inline 展开;
* `Combine Pass`：将一些特定的case进行op的融合，如：
* `Canonicalizer Pass`：通过应用一系列的规则来简化和标准化IR；
* `CSE Pass`：MLIR 的 cse Pass，公共子表达式消除；
* `LICM Pass`：MLIR 的LoopInvariantCodeMotion Pass(https://mlir.llvm.org/doxygen/LoopInvariantCodeMotion_8cpp_source.html)，将循环无关的变量挪到for循环外；


#### 2.2 TritonGPU IR 优化 pass


TritonGPU IR 上的优化主要是 GPU 硬件相关的优化，具体的 Pass如下：

* `ConvertTritonToTritonGPU Pass`：将 Triton IR 转换为 TritonGPU IR，主要是**增加 TritonGPU 硬件相关信息的 layout**；
* `Coalesce Pass`：访存合并pass，重排order，使得最大contiguity 的维度排在最前面
* `Combine Pass`：和Triton IR优化pass功能一致；
* `Pipeline Pass`：MMA 指令（对应功能部件为NVIDIA的Tensor Core）对应的 global memory 到 shared memory 的 N-Buffer 多缓冲优化；
* `Prefetch Pass`：MMA 指令对应的 shared memory 到 register file 的 N-Buffer 多缓冲优化；
* `Canonicalizer`：和Triton IR优化pass功能一致；
* `CSE Pass`：和Triton IR优化pass功能一致；
* `LICM Pass`：和Triton IR优化pass功能一致；



### 3 NVIDIA Triton Server

NVIDIA Triton Server是一个开源的推理服务平台，由NVIDIA公司推出，旨在为各种AI工作负载提供标准化的模型部署和执行。支持在不同处理器（包括GPU、CPU等）上运行基于各种框架（如TensorFlow、PyTorch、ONNX等）训练的模型。它不仅能够实现模型的高效部署，还能保证推理服务的高性能和可扩展性。

NVIDIA Triton Server 支持包括HTTP和gRPC在内的多种通信协议，确保了与不同客户端的兼容性和高效的数据交换。能够与TensorFlow、TensorRT、PyTorch和ONNXRuntime等多种推理引擎后端无缝对接，并且采用了C++作为开发语言，通过C++ API直接与底层的计算引擎交互，从而确保了处理请求时的高性能。

NVIDIA Triton Server 支持多模型的并发执行和动态批处理技术，能显著提升GPU资源的利用率，并优化整个推理服务的性能。多模型集成（ensemble）功能，允许将多个模型作为一个整体进行部署和推理，能够应对需要多个模型协同工作的复杂场景。
