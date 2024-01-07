---
layout: post
title: ai_compiler_basic
date: 2023-12-24
tags: [AI]
author: taot
---

## AI compiler 介绍

* 什么是 AI compiler，为什么需要 AI compiler：深度学习加速编译器

    ![1](https://image-hosting-taot.oss-cn-shanghai.aliyuncs.com/markdown-image202305162126863.png)

* 业界主流 AI compiler

    *   TVM：陈天奇
        *   主推多后端，GPU/TPU/其他芯片
        *   端到端的加速的效果，从输入网络模型，直接给出可在GPU/TPU上运行的 cuda 代码
    *   TensorComprehension（TC）
        *   端到端的加速效果
        *   对 pytorch 支持的比较好
        *   自动生成高性能的机器代码的C++库
        *   基于 Halide，ISL，NVRTC，LLVM

    [知乎：如何看待Tensor Comprehensions？与TVM有何异同？][]

    TVM：各种深度学习框架模型 --> CoreML/ONNX(通用格式) --> NNVM/Relay(神经网络模型层的优化) --> 图优化 --> TVM --> TVM 算子层的优化 

    TVM 输入不同框架，不同版本的深度学习网络模型，输出支持多个平台，多个场上的不同芯片的，高效的代码

    TVM架构图：

    ![TVM架构](https://image-hosting-taot.oss-cn-shanghai.aliyuncs.com/markdown-image202305162159418.png)

    TC 架构图：

    ![TC架构](https://image-hosting-taot.oss-cn-shanghai.aliyuncs.com/markdown-image202305162206371.png)

    两者架构不同，TC不存在图层优化编译；实现的优化不同；TVM中不仅存在支持描述算子的计算，还支持输入输出tensor 定义；TVM 的Schedule 是通过指令用户手工指定；输出Tensor 的shape不同

    *   MindSpore：华为昇腾芯片
        *   针对网络模型输入，生成能够在升腾芯片上高速执行的代码
        *   完善的工具链
        *   auto kernel generator
    *   TensorRT：NVIDIA
    *   Polly LLVM

图层和算子层优化：

*   图层：resnet50，bert
*   算子层：

### TensorFlow 常见算子

