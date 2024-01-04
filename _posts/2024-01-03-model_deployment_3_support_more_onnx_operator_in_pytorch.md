---
layout: post
title: model_deployment_3_support_more_onnx_operator_in_pytorch
date: 2024-01-03
---

## 在 pytorch 中支持更多的 onnx 算子

将 pytorch model 转换成 onnx model，需要满足：
* 算子在 PyTorch 中有实现
* 有把该 PyTorch 算子映射成一个或多个 ONNX 算子的方法
* ONNX 有相应的算子

三个条件都有可能缺失，这三个条件的支持方式：
* 添加PyTorch 算子实现
  * 组合现有算子
  * 添加 TorchScript 算子
  * 添加普通 C++ 拓展算子
* 添加 pytorch 算子到 onnx 算子的映射方法
  * 为 ATen 算子添加符号函数
  * 为 TorchScript 算子添加符号函数
  * 封装成 torch.autograd.Function 并添加符号函数
* 添加 ONNX 算子
  * 使用现有 ONNX 算子
  * 定义新 ONNX 算子

### 1 支持 ATen 算子

当算子在 ATen 中已经实现了，ONNX 中也有相关算子的定义，但是相关算子映射成 ONNX 的规则没有写的时候，只需要**为 ATen 算子补充描述映射规则的符号函数**就可以。

onnx 的 `Asinh` 算子在 Aten 中有相应的函数定义，但是缺少映射到 onnx 算子的符号函数。这里以 `Asinh` 算子来实践为 ATen 算子添加符号函数。

> ATen 是 PyTorch 内置的 C++ 张量计算库，PyTorch 算子在底层绝大多数计算都是用 ATen 实现的。

#### 1.1 获取 ATen 中算子接口定义

首先，需要获得`asinh`推理接口的输入参数定义。在`torch/_C/_VariableFunctions.pyi` 和 `torch/nn/functional.pyi` 这两个文件中搜索相应的算子名称（*一般来说 ATen 中的函数名称都是全小写，因此在搜索算子名的时候尽量忽略大小写*）。这两个文件是编译 PyTorch 时本地自动生成的文件，里面包含了 ATen 算子的 PyTorch 调用接口。搜索结果，`def asinh` 定义在 `torch/_C/_VariableFunctions.pyi`文件中。其接口定义有两个实现：

```python
    def asinh(input: Tensor, *, out: Optional[Tensor]=None) -> Tensor: ... 
    def asinh_(input: Tensor) -> Tensor: ...
```

#### 1.2 添加符号函数

符号函数，可以看成是 PyTorch 算子类的一个静态方法。在把 PyTorch 模型转换成 ONNX 模型时，各个 PyTorch 算子的符号函数会被依次调用，以完成 PyTorch 算子到 ONNX 算子的转换。符号函数的一般定义模板：
```python
    def symbolic(g: torch._C.Graph, input_0: torch._C.Value, input_1: torch._C.Value, ...): 
```
其中，`torch._C.Graph` 和 `torch._C.Value` 都对应 PyTorch 的 C++ 实现里的一些类。第一个参数固定是 g，它表示和计算图相关的内容；后面的每个参数都表示算子的输入，需要和算子的前向推理接口的输入相同。对于 ATen 算子来说，它们的前向推理接口就是前面两个 .pyi 文件里的函数接口。

在把 PyTorch 算子转换成 ONNX 算子时，需要在符号函数中调用 `g.op`方法来为最终的计算图添加一个 ONNX 算子。其定义：
```python
    def op(name: str, input_0: torch._C.Value, input_1: torch._C.Value, ...) 
```
其中，第一个字符串参数是算子名称。如果该算子是普通的 ONNX 算子，只需要把它在 ONNX 官方文档里的名称填进去即可。

当 pytorch 算子比较简单的时候，可以直接映射为 onnx 的某一个算子， 此时只需要把 PyTorch 算子的输入用`g.op()`一一对应到 ONNX 算子上，并把 `g.op()`的返回值作为符号函数的返回值即可；当 pytorch 算子比较复杂的时候，没法直接映射到某一个 onnx 算子，这个时候就需要映射到多个 onnx 算子。

在 [Asinh 算子的官方文档](https://link.zhihu.com/?target=https%3A//github.com/onnx/onnx/blob/main/docs/Operators.md%23asinh)中，可以看到该算子的描述：一个输入 input，一个输出 output，二者的类型都为张量。

在需要导出包含 Asinh 算子的网络的位置，添加代码：
```python
    from torch.onnx.symbolic_registry import register_op 
 
    def asinh_symbolic(g, input, *, out=None): 
        return g.op("Asinh", input) 
     
    register_op('asinh', asinh_symbolic, '', 9)  
```
这里的asinh_symbolic就是asinh的符号函数。从除g以外的第二个输入参数开始，其输入参数应该严格对应它在 ATen 中的定义。

在符号函数的函数体中，`g.op("Asinh", input)`完成了 ONNX 算子的定义。其中，第一个参数`"Asinh"`是算子在 ONNX 中的名称；第二个参数 input，与官方文档中的描述一致，这个算子只有一个输入，因此只要把符号函数的输入参数 input 对应过去就行。ONNX 的 Asinh 的输出和 ATen 的 asinh 的输出是一致的，因此我们直接把 g.op() 的结果返回即可。

定义完符号函数后，要把这个符号函数和原来的 ATen 算子**绑定**起来。这里，要用到 `register_op` 这个 PyTorch API 来完成绑定：`register_op('asinh', asinh_symbolic, '', 9) `。`register_op`的第一个参数是目标 ATen 算子名; 第二个是要注册的符号函数; 第三个参数是算子的**域**，对于普通 ONNX 算子，直接填空字符串即可; 第四个参数表示向哪个算子集版本注册。另外，这里向第 9 号算子集注册，不代表较新的算子集（第 10 号、第 11 号……）都得到了注册。

包含单算子 Asinh 的 pytorch model 转成 onnx model：
```python
    import torch 
 
    class Model(torch.nn.Module): 
        def __init__(self): 
            super().__init__() 
     
        def forward(self, x): 
            return torch.asinh(x) 
     
    from torch.onnx.symbolic_registry import register_op 
     
    def asinh_symbolic(g, input, *, out=None): 
        return g.op("Asinh", input) 
     
    register_op('asinh', asinh_symbolic, '', 9) 
     
    model = Model() 
    input = torch.rand(1, 3, 10, 10) 
    torch.onnx.export(model, input, 'asinh.onnx') 
    
    # 测试新添加的算子正确性
    # 一般用 PyTorch 运行一遍原算子，再用推理引擎（比如 ONNX Runtime）运行一下 ONNX 算子，最后比对两次的运行结果
    import onnxruntime 
    import numpy as np 

    torch_output = model(input).detach().numpy() 
 
    sess = onnxruntime.InferenceSession('asinh.onnx') 
    ort_output = sess.run(None, {'0': input.numpy()})[0] 
     
    assert np.allclose(torch_output, ort_output) 
```


### 2 支持 TorchScript 算子

对于一些比较复杂的运算，PyTorch 原生算子无法满足功能。此时需要添加自定义 pytorch 算子，再将其转换成 onnx。[TorchScript 算子](https://pytorch.org/tutorials/advanced/torch_script_custom_ops.html)。

此处以可变形卷积（Deformable Convolution）算子为例，记录为现有 TorchScript 算子添加 ONNX 支持的方法。

为算子添加符号函数的一般流程：
* 获取原算子的前向推理接口。
* 获取目标 ONNX 算子的定义。
* 编写符号函数并绑定。

#### 2.1 使用 TorchScript 算子

* 定义一个包含 DeformConv2d 算子的模型：
```pyhton
    import torch 
    import torchvision 
     
    class Model(torch.nn.Module): 
        def __init__(self): 
            super().__init__() 
            self.conv1 = torch.nn.Conv2d(3, 18, 3) 
            self.conv2 = torchvision.ops.DeformConv2d(3, 3, 3) 
     
        def forward(self, x): 
            return self.conv2(x, self.conv1(x)) 
```

* 获取 DeformConv2d 算子的前向推理接口
  ` torchvision/csrc/ops/deform_conv2d.cpp ` 中的 `deform_conv2d` 算子的调用接口：
  ```cpp
      m.def(TORCH_SELECTIVE_SCHEMA( 
        "torchvision::deform_conv2d(Tensor input,  
        Tensor weight,  
        Tensor offset,  
        ...... 
        bool use_mask) -> Tensor")); 
  ```

* 获取目标 ONNX 算子的定义
  目前 onnx 还没有支持 deform_conv2d 算子，因此在官方文档中也查不到 deform_conv2d 算子相关的定义。

#### 2.2 自定义 onnx 算子

onnx 没有支持的 onnx 算子，需要自定义该 onnx 算子。`g.op()` 函数可以用来自定义 onnx 算子的函数。对于 onnx 官方支持的算子，使用 `g.op()`时候的第一个参数是该 onnx 算子的名称；对于自定义算子，`g.op()`的第一个参数是一个带命名空间的算子名，例如：
```python
    g.op("custom::deform_conv2d") 
```
其中，命名空间是为了防止命名冲突，如果在 `g.op()` 里不加前面的命名空间，则算子会被默认成 ONNX 的官方算子。

PyTorch 在运行 `g.op()` 时会对官方的算子做检查，如果算子名有误，或者算子的输入类型不正确， `g.op()` 就会报错。

新自定义算子的符号函数：
```python
    @parse_args("v", "v", "v", "v", "v", "i", "i", "i", "i", "i", "i", "i", "i", "none") 
    def symbolic(g,  
            input, 
            weight, 
            offset, 
            mask, 
            bias, 
            stride_h, stride_w, 
            pad_h, pad_w, 
            dil_h, dil_w, 
            n_weight_grps, 
            n_offset_grps, 
            use_mask): 
        return g.op("custom::deform_conv2d", input, offset) 
```
符号函数的参数来自于`deform_conv2d` 算子的调用接口函数。

关于装饰器 `@parse_args` 的说明。TorchScript 算子的符号函数要求标注出每一个输入参数的类型。比如"v"表示 Torch 库里的 value 类型，一般用于标注张量，而"i"表示 int 类型，"f"表示 float 类型，"none"表示该参数为空。具体描述可以在 `torch.onnx.symbolic_helper.py` 里面看到。这里输入参数中的 input, weight, offset, mask, bias 都是张量，所以用"v"表示。后面的其他参数同理。

注册符号函数：
```python
    register_custom_op_symbolic("torchvision::deform_conv2d", symbolic, 9) 
```
和前面的 register_op 类似，注册符号函数时，要输入算子名、符号函数、算子集版本。与前面不同的是，这里的算子集版本是最早生效版本，在这里设定版本 9，意味着之后的第 10 号、第 11 号……版本集都能使用这个新算子。


### 3 使用 torch.autograd.Function

用 `torch.autograd.Function` 封装新算子，可以很简单地为 PyTorch 添加 C++ 算子实现。

`torch.autograd.Function` 能完成算子实现和算子调用的隔离。不管算子是怎么实现的，它封装后的使用体验以及 ONNX 导出方法会和原生的 PyTorch 算子一样。

这里以一个自定义算子 `my_add`为例，这个算子输入张量 a, b ，输出 2a + b 的值。


#### 3.1 为 PyTorch 添加 C++ 拓展

对于自定义算子 my_add，可以用以下的 C++ 源文件来实现。该文件命名为 "my_add.cpp"：
```cpp
    // my_add.cpp 
 
    #include <torch/torch.h> 
     
    torch::Tensor my_add(torch::Tensor a, torch::Tensor b) 
    { 
        return 2 * a + b; 
    } 
     
    PYBIND11_MODULE(my_lib, m) 
    { 
        m.def("my_add", my_add); 
    } 
```
这里，torch::Tensor 是 C++ 中 torch 的张量类型，它的加法和乘法等运算符均已重载。因此，可以像对普通标量一样对张量做加法和乘法。

用 `PYBIND11_MODULE` 来为 C++ 函数提供 Python 调用接口。这里的 my_lib 是接下来要在 Python 里导入的模块名。双引号中的 my_add 是 Python 调用接口的名称，这里对齐 C++ 函数的名称，依然用 "my_add"这个名字。

编写如下的 Python 代码并命名为 "setup.py"，来编译刚刚的 C++ 文件：
```python
    from setuptools import setup 
    from torch.utils import cpp_extension 
     
    setup(name='my_add', 
          ext_modules=[cpp_extension.CppExtension('my_lib', ['my_add.cpp'])], 
          cmdclass={'build_ext': cpp_extension.BuildExtension}) 
```

这里使用 Python 的 setuptools 编译功能和 PyTorch 的 C++ 拓展工具函数，可以编译包含 torch 库的 C++ 源文件。这里我们填写模块名 `my_lib` 和模块中的源文件名 `my_add.cpp`。

执行编译命令：`python setup.py develop `


#### 3.2 用 torch.autograd.Function 封装

使用 `torch.autograd.Function` 来封装算子的底层调用：
```python
    import torch 
    import my_lib 
    class MyAddFunction(torch.autograd.Function): 
     
        @staticmethod 
        def forward(ctx, a, b): 
            return my_lib.my_add(a, b) 
     
        @staticmethod 
        def symbolic(g, a, b): 
            two = g.op("Constant", value_t=torch.tensor([2])) 
            a = g.op('Mul', a, two) 
            return g.op('Add', a, b) 
```
Function 类本身表示 PyTorch 的一个可导函数，只要为其定义了前向推理和反向传播的实现，就可以把它当成一个普通 PyTorch 函数来使用。

PyTorch 会自动调度该函数，合适地执行前向和反向计算。Function 类有一个很好的性质：如果它定义了 symbolic 静态方法，该 Function 在执行 torch.onnx.export() 时就可以根据 symbolic 中定义的规则转换成 ONNX 算子。这里的 synbolic 函数就是符号函数，但是其名称为 `symbolic`

在 forward 函数中，用 my_lib.my_add(a, b) 就可以调用之前写的C++函数了。这里 my_lib 是库名，my_add 是函数名，这两个名字是在前面C++的 PYBIND11_MODULE 中定义的。

在 symbolic 函数中，用 g.op() 定义了三个算子：常量、乘法、加法。这里乘法和加法只需要根据 ONNX 算子定义规则把输入参数填入即可。而在定义常量算子时，要把 PyTorch 张量的值传入 value_t 参数中。

在 ONNX 中，需要把新建常量当成一个算子来看待，尽管这个算子并不会以节点的形式出现在 ONNX 模型的可视化结果里。

把算子封装成 Function 后，就可以把 my_add算子用起来了：
```python
    my_add = MyAddFunction.apply 
 
    class MyAdd(torch.nn.Module): 
        def __init__(self): 
            super().__init__() 
     
        def forward(self, a, b): 
            return my_add(a, b) 
```
这里，apply 是 torch.autograd.Function 的一个方法，这个方法完成了 Function 在前向推理或者反向传播时的调度。在使用 Function 的派生类做推理时，不应该显式地调用 forward()，而应该调用其 apply 方法。

用下面的代码来导出一个包含新算子的 ONNX 模型，并验证一下它是否正确：
```python
    model = MyAdd() 
    input = torch.rand(1, 3, 10, 10) 
    torch.onnx.export(model, (input, input), 'my_add.onnx') 
    torch_output = model(input, input).detach().numpy() 
     
    import onnxruntime 
    import numpy as np 
    sess = onnxruntime.InferenceSession('my_add.onnx') 
    ort_output = sess.run(None, {'a': input.numpy(), 'b': input.numpy()})[0] 
     
    assert np.allclose(torch_output, ort_output) 
```
