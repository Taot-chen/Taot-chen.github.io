---
layout: post
title: torch_source_code_2_torch_dynamo_optimize
date: 2024-06-25
tags: [torch]
author: taot
---

## pytorch 源码阅读(2)——torch._dynamo.optimize

### 0 `torch._dynamo.optimize(backend, *, nopython, guard_export_fn, guard_fail_fn, disable, dynamic)`，TorchDynamo 的主入口点

### 1 参数说明

* backend，一般有两种情况：
   *  一个包含 torch.fx.GraphModule 和 example_inputs，返回一个能够更快执行 graph 的函数或者可调用的对象。也可以通过设置backend_ctx_ctor 属性，来给后端提供额外的上下文。具体用法可以查看`AOTAutogradMemoryEfficientFusionWithContext`。
   *  或者是一个`torch._dynamo.list_backends()`里面的字符串后端名称。
* nopython: 如果时 True，graph breaks 将会报错，并且只有一个完整的 graph
* disable：如果为 True，设置当前装饰器为 no-op
* dynamic：(bool or None): 使用动态 shape 进行 tracing
  * True：尝试生成尽可能动态的 kernel，但这个也不总是有效的，因为某些特化优化会使动态 kernel 失效
  * False：将不生成任何动态的 kernel，只进行特化优化
  * None：默认值，会自动检测是否发生了动态变化，并在重新编译时编译一个更动态的动态内核。

### 2 额外说明

这个函数时 TorchDynamo 的主入口点。 进行 graph 提取并调用 backend() 来优化提取到的 graph。
