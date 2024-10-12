---
layout: post
title: torch_source_code_1_torch_compiler
date: 2024-06-25
tags: [torch]
author: taot
---

## pytorch 源码阅读(1)——torch.complie

### 0  `torch.complie(model, fullgraph, dynamic, backend, mode, options, disable)`，使用 TorchDynamo 和指定的后端优化给定的模型/函数

`torch.compiler`是一个命名空间，通过它可以将一些内部编译器方法公开给用户使用。此命名空间中的主要函数和功能是`torch.compile`。

`torch.compile`是`PyTorch 2.x`中引入的 PyTorch 函数，旨在解决 PyTorch 中的准确图形捕获问题，并最终使软件工程师能够更快地运行其 PyTorch 程序。

对于在编译区域内执行的每一帧，都会尝试编译，并将编译结果缓存到代码对象中，供将来使用。 如果之前的编译结果不适用于后续调用那么单帧可能会被编译多次，这被称为 `guard_failure`，此时可以使用 `TORCH_LOGS=guards` 来进行调试。

> *TorchDynamo 是一个设计用于加速未修改的 PyTorch 程序的 Python 级即时（JIT）编译器。它通过 Python Frame Evaluation Hooks（Python 框架评估钩子）来实现这一目标，以便在运行时动态地生成和优化代码。这使得 TorchDynamo 可以有效地处理各种 Python 代码，包括包含控制流（如循环和条件语句）的代码，而无需进行任何修改。*

### 1 参数说明

`torch.complie(model, fullgraph, dynamic, backend, mode, options, disable)`

* model(callable)：需要进行优化的模型或者函数
* fullgraph(bool)：是否可以将模型分解为多个子图
* dynamic (bool or None): 使用动态 shape 进行 tracing
  * True：尝试生成尽可能动态的 kernel，但这个也不总是有效的，因为某些特化优化会使动态 kernel 失效
  * False：将不生成任何动态的 kernel，只进行特化优化
  * None：默认值，会自动检测是否发生了动态变化，并在重新编译时编译一个更动态的动态内核。
* backend (str or Callable)：使用的后端
  * inductor，默认的后端，在性能和开销之间有较好的平衡
  * 可以使用 `torch._dynamo.list_backends()` 查看非实验性树内后端
  * 可使用 `torch._dynamo.list_backends(None)` 查看实验性或调试性的树内后端
  * 注册树外后端的方式：https://pytorch.org/docs/main/compile/custom-backends.html
* mode(str)，可以配置的值："default", "reduce-overhead", "max-autotune" or "max-autotune-no-cudagraphs"
  * default：默认模式，在性能和开销之间进行很好的平衡
  * reduce-overhead：降低开销模式，小 batch 的场景比较有用，降低 python 使用 CUDA garphs 的开销。降低开销的代价是使用更多的内存，因为会缓存调用所需的工作区内存，这样就在后续运行中无需重新分配。减少开销并不能保证一定有效，只是减少了不改变输入的 CUDA graphs 的开销。在 CUDA graphs 不适用的其他情况下，可以使用 TORCH_LOG=perf_hints 来进行 debug。
  * max-autotune：利用基于 Triton 的矩阵乘法和卷积的模，默认开启 CUDA graphs。
  * max-autotune-no-cudagraphs：与 max-autotune 模式类似，但是不开启 CUDA graphs。
  * 调用 `torch._inductor.list_mode_options()` 可以查看每个模式的具体配置。
* options (dict): 传递给后端的一些字典选项，一些值得尝试的配置：
  * `epilogue_fusion`，将点对点融合设置到模板中，需要同时设置`max_autotune`
  * `max_autotune`，通过配置文件选择最佳的 matmul 配置
  * `fallback_random`，在调试精度问题的时候比较有用
  * `shape_padding`，填充矩阵形状，以便更好地调整 GPU 上的负载，尤其是张量内核的负载
  * `triton.cudagraphs`，可减少 Python 在使用 CUDA graphs 时的开销
  * `trace.enabled`，最好用的 debug 选项，建议打开
  * `trace.graph_diagram`，将显示融合后的 graph 的图片
  * 可以通过调用 `torch._inductor.list_options()` 查看支持的全部配置列表
* disable (bool): 将 torch.compile 设置为 no-op 的测试模式

### 2 额外说明

* 这个函数一开始有这么一行：`_C._log_api_usage_once("torch.compile")`，这个是用来进行调试的。
  * `_C._log_api_usage_once("torch.compile")`，这是严格意义上的调试，很可能大部分使用都是在 facebook 内部进行的。如果设置环境变量`PYTORCH_API_USAGE_STDERR=1`，就能看到 PyTorch 记录了 "涉及 PyTorch 的哪些部分"，这很可能会在出现意外情况时大致知道该去哪里找（以及找谁）。这个函数出现的原因是 pytorch 现在的规模已经很多大了，往往会出现很多意想不到的交互调用，这个函数可以帮助缩小 debug 的范围。

* 这个函数的作用是使用 TorchDynamo 和指定的后端优化给定的模型/函数，进行包装和参数配置的设置，最终会调用`torch._dynamo.optimize`进行优化处理。

### 3 torch.compile 用到的底层技术

* **TorchDynamo (torch._dynamo)** 是一个内部 API，它使用名为 Frame Evaluation API 的 CPython 功能来安全地捕获 PyTorch 图形。可供 PyTorch 用户外部使用的函数通过 torch.compiler 命名空间公开。

* **TorchInductor** 是默认的 torch.compile 深度学习编译器，可为多个加速器和后端生成快速代码。需要使用后端编译器才能通过 torch.compile 实现加速。对于 NVIDIA 和 AMD GPU，它利用 OpenAI Triton 作为关键构建模块。

* **AOT Autograd** 不仅捕获用户级代码，还捕获反向传播，从而导致“提前”捕获反向传递。这使得能够使用 TorchInductor 加速正向和反向传递。


如上所述，为了更快地运行工作流，torch.compile 通过 TorchDynamo 需要一个后端，该后端将捕获的图形转换为快速的机器代码。不同的后端可以产生不同的优化收益。默认后端称为`TorchInductor`，也称为`inductor`。可以通过运行`torch.compiler.list_backends()`查看支持哪些后端，每个后端都有其可选依赖项。

**一些最常用的后端**
  * 训练和推理后端
    |后端|说明|
    |---|---|
    |`torch.compile(m, backend="inductor")`|使用 TorchInductor 后端。|
    |`torch.compile(m, backend="cudagraphs")`|具有 AOT Autograd 的 CUDA 图形。|
    |`torch.compile(m, backend="ipex")`|在 CPU 上使用 IPEX。|
    |`torch.compile(m, backend="onnxrt")`|在 CPU/GPU 上使用 ONNX Runtime 进行训练。|

  * 仅推理后端
    |后端|说明|
    |---|---|
    |`torch.compile(m, backend="tensorrt")`|使用 Torch-TensorRT 进行推理优化。需要在调用脚本中 import torch_tensorrt 来注册后端。|
    |`torch.compile(m, backend="ipex")`|在 CPU 上使用 IPEX 进行推理。|
    |`torch.compile(m, backend="tvm")`|使用 Apache TVM 进行推理优化。|
    |`torch.compile(m, backend="openvino")")`|使用 OpenVINO 进行推理优化。|
