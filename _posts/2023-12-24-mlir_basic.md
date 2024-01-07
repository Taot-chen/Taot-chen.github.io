---
layout: post
title: mlir_basic
date: 2023-12-24
tags: [mlir]
author: taot
---

## MLIR 初学

*   什么是 MLIR 
*   MLIR Toy Example 
*   Dialects
*   使用 MLIR 的框架

### 1、motivation

*   AI compiler：输入网络模型，转成IR，在IR上进行各种优化，高级 IR 翻译成低级 IR，最后低级 IR 分别转换成多后端
*   前面的高级 IR (可能有多层，multi level IR)，高级 IR 转成的低级 IR 转化到 LLVM 上 ，再借助 LLVM 强大的功能，分配到实际的硬件的 backend 上

![AICompiler1](https://image-hosting-taot.oss-cn-shanghai.aliyuncs.com/markdown-image202305162237662.png)

*   MLIR 是一个可重用的编译器的工具盒。MLIR 允许通过多层不同的 IR 分别根据某一层或者某一点做的优化，去设计对应的 IR 
*   MLIR 能够允许在一个比较大的 OP（如reduce算子）里面同时存在多个不同的 dialect
*   避免过早的 lowering 导致的信息丢失。优化过早容易丢失信息，过晚容易导致复杂度飙升

### 2、Toy Example

![Toy_example](https://image-hosting-taot.oss-cn-shanghai.aliyuncs.com/markdown-image202305170107608.png)

dialects

Linalg dialect

attine dialect，表达控制流

sfc dialect，表达控制流

STD dialect，standard 调用

Memref dialect，一个结构体，存放要处理的内存的必要信息

### 3、使用 MLIR 的框架

*   TensorFlow
*   Torch-MLIR
*   ONNX-MLIR

### 1、MLIR 简介

常见的 IR 表示系统：

![常见IR表示](https://image-hosting-taot.oss-cn-shanghai.aliyuncs.com/markdown-image202305170140830.jpg)

Clang 对 AST 进行静态分析和转换操作，各个语言的 AST 都需要进行类似的优化转换成对应的 IR

深度学习框架 --> 图 IR --> 转换成某个后端对应的 IR

问题：

*   IR 种类太多，不同 IR 的同类 pass 不兼容，针对新的 IR 编写 pass 需要重新学习 IR 语法
*   不同种类的 IR 所做的 pass 优化在下一层不可见
*   不同类型的 IR 之间的转换开销大，如 图 IR 到 LLVM IR 转换开销大

TensorFlow 框架常见的 IR：

![TF_IR_MLIR](https://image-hosting-taot.oss-cn-shanghai.aliyuncs.com/markdown-image202305170156621.jpg)

使用 dialect 构建 IR 表示系统：

![dialect](https://image-hosting-taot.oss-cn-shanghai.aliyuncs.com/markdown-image202305170158532.jpg)

dialect 可以简单看作是 **具有 IR 表示能力的黑盒子**

### 2、Toy 接入  MLIR

Toy 语言：为了验证及演示 MLIR 系统的真个流程而开发的一种基于 TensorFlow 的语言

MLIR 表达式的生成：

​	Toy 源程序 --> Toy AST --> Toy dialect（遍历 AST，根据节点生成对应的 operation） --> Toy IR  MLIR 表达式 --> Lowered MLIR 表达式 --> LLVM IR  --> 目标程序

![MLIR表达式的生成](https://image-hosting-taot.oss-cn-shanghai.aliyuncs.com/markdown-image202305170213370.png)

MLIR 表达式

### 3、Dialect 及 Operation 详解

一个 MLIR dialect 包含：

	* 一个命名空间（namespace），表示当前 dialect 的名称
	* 一个自定义类型，可有可无，每一个都是一个 C++ class
	* 一组 operation，mlir 中的核心，类似于 LLVM 中的 instruction
	* 可能有的解析器和打印器，针对当前的 MLIR 进行一些信息的解析和打印
	* passes：对 dialect 的优化，如分析、转换、dialect 之间的转换

translation、conversion、transformation的区别：

![dialect2](https://image-hosting-taot.oss-cn-shanghai.aliyuncs.com/markdown-image202305170223163.jpg)

MLIR 重要构件，operation

![operation](https://image-hosting-taot.oss-cn-shanghai.aliyuncs.com/markdown-image202305170224669.jpg)

两个基于 tableGen 的模块：

*   ods：用来帮助创建 operation，操作定义的规范，在.td 中定义好必要的自带内容，使用 mlir 的 tableGen 工具，就可以自动生成对应的 operation 的 C++ class 代码
*   drr：声明性规则重写

整个 tableGen 模块是基于 operation definition specification（ODS）框架进行编写以及发挥作用

*   dialect 链接 --> 定义基类 --> 自定义 operation
*   使用 mlir-tablegen 工具，生成 C++ 文件
*   开发的单源性，避免冗余开发；促进自动化生成，避免手动开发 operation

### 4、MLIR 表达式变形

*   手动编写代码进行表达式的匹配与重写
*   DDR（declaration rewrite rules）使用基于规则的方式来自动生成表达式匹配和重写函数

### 5、Lowering 过程（Dialect  Conversion）

把 MLIR 表达式逐渐降级成低层次的 Dialect 抽象，for：更加贴近硬件，进行代码生成；做硬件相关的优化

![lowering](https://image-hosting-taot.oss-cn-shanghai.aliyuncs.com/markdown-image202305170256279.jpg)

dialect 之间的转换

三个重要的步骤：

*   conversion target：明确要转换的目标，对要转换的dialect进行合法化
*   operation conversion
*   type conversion

pattern match and rewrite：

一个高抽象级别的 dialect 到一个低抽象级别的 dialect 过程中，可以只 lowering 其中一部分 operation，剩下的 operation 只需要升级与其他 operation 共存

full lowering：需要把各种 dialect 全部 lowering 到 LLVM dialect ，再 lowering 到 LLVM IR，接入到 LLVM 后端进行 CodeGen

*   创建 lower to LLVM Pass
*   type conversion
*   conversion target
*   定义 lowering pattern

![full_lowering](https://image-hosting-taot.oss-cn-shanghai.aliyuncs.com/markdown-image202305170311231.jpg)

dialect 早期体系：

![dialect早期体系](https://image-hosting-taot.oss-cn-shanghai.aliyuncs.com/markdown-image202305170313199.jpg)

