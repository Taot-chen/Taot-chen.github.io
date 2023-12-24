---
layout: post
title: llvm_basic
date: 2023-12-24
---

## LLVM 初识

compiler：make source code to binary

GCC 被用来解决现实的问题，因此没有精力注重架构的完美设计

LLVM 和 GCC 的区别

*   LLVM 是一个 compiler
*   LLVM 是一个 compiler 平台
*   LLVM 是一系列的编译器工具
*   LLVM 是编译工具链
*   LLVM 是开源的 C++ implementation

*   GCC 支持更多的语言：C++，Java，Ada，Fortrtan，Go 等
*   GCC 比 LLVM 支持的 CPU 更多
*   GCC 支持许多语言拓展

Compiler：

​	source code --> (compiler) --> executable binary

​	source code --> (front end --> optimization --> backend) --> machine code

optimization:

*   SSA(Static single Assignment)
*   constant propagation：常量释放
*   dead code elimination：无用 code 清理
*   branch free：

instruction pipeline

backend

![LLVM1](https://image-hosting-taot.oss-cn-shanghai.aliyuncs.com/markdown-image202305171547408.png)



### 使用 LLVM MLIR 设计编译器

