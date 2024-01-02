---
layout: post
title: model_deployment_2_torch2onnx
date: 2024-01-02
---

## pytorch 转 onnx
ONNX 是目前模型部署中最重要的中间表示之一，在把 PyTorch 模型转换成 ONNX 模型时，使用的 torch 接口是 `torch.onnx.export`
这里记录了 pytorch 模型转 onnx 时的原理和注意事项，还包括部分 PyTorch 与 ONNX 的算子对应关系。

### 1 `torch.onnx.export`原理
#### 1.1 导出计算图
TorchScript 是一种序列化和优化 PyTorch 模型的格式，在优化过程中，一个torch.nn.Module模型会被转换成 TorchScript 的 torch.jit.ScriptModule 模型。通常 TorchScript 也被当成一种中间表示来使用。

torch.onnx.export中需要的模型实际上是一个torch.jit.ScriptModule。而要把普通 PyTorch 模型转一个这样的 TorchScript 模型，有跟踪（trace）和记录（script）两种导出计算图的方法。

如果给torch.onnx.export传入了一个普通 PyTorch 模型（torch.nn.Module），那么这个模型会默认使用 trace 的方法导出：
$$\boxed{torch.nn.Module} \xrightarrow{torch.onnx.export（默认使用 torch.jit.trace）} \boxed{onnx模型}$$
$$\boxed{torch.nn.Module} \xrightarrow[torch.jit.scripts]{torch.jit.trace} \boxed{torch.jit.ScriptModule} \xrightarrow{torch.onnx.export} \boxed{onnx模型}$$

trace 方法只能通过实际运行一遍模型的方法导出模型的静态图，即无法识别出模型中的控制流（如循环）；script 方法则能通过解析模型来正确记录所有的控制流

下面的代码段额可以用来对比 trace 和 script 两种方法获取 graph 的区别
```python
    import torch 
 
    class Model(torch.nn.Module): 
        def __init__(self, n): 
            super().__init__() 
            self.n = n 
            self.conv = torch.nn.Conv2d(3, 3, 3) 
    
        def forward(self, x): 
            for i in range(self.n): 
                x = self.conv(x) 
            return x 
    
    
    models = [Model(2), Model(3)] 
    model_names = ['model_2', 'model_3'] 
    
    for model, model_name in zip(models, model_names): 
        dummy_input = torch.rand(1, 3, 10, 10) 
        dummy_output = model(dummy_input) 
        model_trace = torch.jit.trace(model, dummy_input) 
        model_script = torch.jit.script(model) 
    
        # 跟踪法与直接 torch.onnx.export(model, ...)等价 
        torch.onnx.export(model_trace, dummy_input, f'{model_name}_trace.onnx', example_outputs=dummy_output) 
        # 记录法必须先调用 torch.jit.sciprt 
        torch.onnx.export(model_script, dummy_input, f'{model_name}_script.onnx', example_outputs=dummy_output) 
```
在这段代码里，定义了一个带循环的模型，模型通过参数n来控制输入张量被卷积的次数。之后，各创建了一个n=2和n=3的模型。把这两个模型分别用跟踪和记录的方法进行导出。

值得一提的是，由于这里的两个模型（model_trace, model_script）是 TorchScript 模型，export函数已经不需要再运行一遍模型了。（如果模型是用跟踪法得到的，那么在执行torch.jit.trace的时候就运行过一遍了；而用记录法导出时，模型不需要实际运行）参数中的dummy_input和dummy_output`仅仅是为了获取输入和输出张量的类型和形状。

trace 方法得到的 ONNX 模型结构，会把 for 循环展开，这样不同的 n，得到的 ONNX 模型 graph 是不一样的；而 scripts 方法得到的 ONNX 模型，用 Loop 节点来表示循环，这样对于不同的 n，得到的 ONNX 模型结构是一样的

实际上，推理引擎对静态图的支持更好，通常在模型部署时不需要显式地把 PyTorch 模型转成 TorchScript 模型，直接把 PyTorch 模型用 torch.onnx.export 借助 trace 方法导出即可。

#### 1.2 `torch.onnx.export` 参数注解

这里主要记录对于模型部署比较重要的几个参数在模型部署中还如何设置，该函数的 API 文档：https://pytorch.org/docs/stable/onnx.html#functions

`torch.onnx.export` 在 `torch.onnx.__init__.py` 文件中的定义如下:
```python
    def export(model, args, f, export_params=True, verbose=False, training=TrainingMode.EVAL, 
                input_names=None, output_names=None, aten=False, export_raw_ir=False, 
                operator_export_type=None, opset_version=None, _retain_param_name=True, 
                do_constant_folding=True, example_outputs=None, strip_doc_string=True, 
                dynamic_axes=None, keep_initializers_as_inputs=None, custom_opsets=None, 
                enable_onnx_checker=True, use_external_data_format=False): 
```
前三个必选参数分别为 torch 模型、模型输入（转 ONNX 的时候的 dummy input）、ONNX 模型的保存路径。

* export_params
  模型中是否存储模型权重。IR 中一般包含两类信息，模型结构和模型权重，这两类信息可以在同一个文件里存储，也可以分文件存储。
  一般来说，如果转 onnx 是用来部署的，那么选择设置为 true，存放在同一个文件中；如果是用来在在不同框架间传递模型，则设为 false，分开存放

* input_names, output_names
  设置输入输出张量的名称。如果不设置，会默认使用 tensor ID（数字） 作为 张量名称。ONNX 的张量名称一般都需要设置，因为大部分推理引擎在设置模型输入和获取输出数据的时候，都是以字典的形式进行访问处理，其中张量名称作为 key，数据作为 value

* opset_version
  转换时参考哪个 ONNX 算子集版本，默认为 9

* dynamic_axes
  指定 onnx 动态 shape 的动态维度。
  为了追求效率，ONNX 默认所有参与运算的张量都是静态的（张量的形状不发生改变）。但在实际应用中，我们又希望模型的输入张量是动态的，尤其是本来就没有形状限制的全卷积模型。因此，我们需要显式地指明输入输出张量的哪几个维度的大小是可变的。
  ```python
    import torch 
 
    class Model(torch.nn.Module): 
        def __init__(self): 
            super().__init__() 
            self.conv = torch.nn.Conv2d(3, 3, 3) 
     
        def forward(self, x): 
            x = self.conv(x) 
            return x 
     
     
    model = Model() 
    dummy_input = torch.rand(1, 3, 10, 10) 
    model_names = ['model_static.onnx',  
    'model_dynamic_0.onnx',  
    'model_dynamic_23.onnx'] 
     
    dynamic_axes_0 = { 
        'in' : {0， 'batch'}， 
        'out' : {0, 'batch'} 
    } 
     
    torch.onnx.export(
        model, 
        dummy_input, 
        model_names[0], 
        input_names=['in'], 
        output_names=['out']
    ) 

    torch.onnx.export(
        model, dummy_input, 
        model_names[1], 
        input_names=['in'], 
        output_names=['out'], 
        dynamic_axes=dynamic_axes_0
    ) 
  ```
  导出 2 个 ONNX 模型，分别为没有动态维度、第 0 维动态的模型。
  这里使用字典的方式来表示动态维度，因为  ONNX 要求每个动态维度都有一个名字，否则会有一堆 warning

### 2 torch 转 onnx 时候的额外操作
#### 2.1 添加额外处理逻辑到 onnx 中
可以把一些后处理的逻辑放在模型里，来简化除运行模型之外的其他代码。`torch.onnx.is_in_onnx_export()` 可以达到这样的效果，这个函数只会在执行`torch.onnx.export()`的时候返回 true
```python
    import torch 
 
    class Model(torch.nn.Module): 
        def __init__(self): 
            super().__init__() 
            self.conv = torch.nn.Conv2d(3, 3, 3) 
     
        def forward(self, x): 
            x = self.conv(x) 
            if torch.onnx.is_in_onnx_export(): 
                x = torch.clip(x, 0, 1) 
            return x 
```
这里，仅在模型导出 onnx 时把输出张量的数值限制在[0, 1]之间。使用 `is_in_onnx_export`确让我们方便地在代码中添加和模型部署相关的逻辑。但是，这些突兀的部署逻辑会降低代码整体的可读性。另外，`is_in_onnx_export`只能在每个需要添加部署逻辑的地方都“打补丁”，不方便进行统一的管理。

#### 2.2 中断张量 trace
如果在 pytorch 的模型脚本中有一些比较离谱的操作，会把某些取决于输入的中间结果变成常量，从而使导出的 ONNX 模型和原来的模型不等价。
如下是一个 trace 中断的例子：
```python
    class Model(torch.nn.Module): 
        def __init__(self): 
            super().__init__() 
     
        def forward(self, x): 
            x = x * x[0].item() 
            return x, torch.Tensor([i for i in x]) 
     
    model = Model()       
    dummy_input = torch.rand(10) 
    torch.onnx.export(model, dummy_input, 'a.onnx') 
```
在导出 ONNX 的时候，会有很多的 warning，并且提示转出来的 onnx 很可能不正确。
在这个模型里使用了.item()把 torch 中的张量转换成了普通的 Python 变量，还尝试遍历 torch 张量，并用一个列表新建一个 torch 张量。这些涉及张量与普通变量转换的逻辑都会导致最终的 ONNX 模型不太正确。
另一方面，也可以利用这个性质，在保证正确性的前提下令模型的中间结果变成常量。这个技巧常常用于模型的静态化上。

### 3 pytorch 对 onnx 算子的支持了解
在做 pytorch model 转换成 onnx model 的时候，PyTorch 一方面会用跟踪法执行前向推理，把遇到的算子整合成计算图；另一方面，PyTorch 还会把遇到的每个算子翻译成 ONNX 中定义的算子。 PyTorch 算子是向 ONNX 对齐的，这个过程中，可能会有这样的情况：
* 该算子可以一对一地翻译成一个 ONNX 算子。
* 该算子在 ONNX 中没有直接对应的算子，会翻译成一至多个 ONNX 算子。
* 该算子没有定义翻译成 ONNX 的规则，报错。

#### 3.1 onnx 算子文档
在[onnx 官方算子文档](https://github.com/onnx/onnx/blob/main/docs/Operators.md)中可以查看 onnx 算子的定义情况。
在算子文档中，第一列是算子名，第二列是该算子发生变动的算子集版本号，也就是前面在torch.onnx.export中提到的opset_version表示的算子集版本号。通过查看算子第一次发生变动的版本号，可以知道某个算子是从哪个版本开始支持的；通过查看某算子小于等于opset_version的第一个改动记录，可以知道当前算子集版本中该算子的定义规则。

#### 3.2 pytorch 对 onnx 算子的映射
在 PyTorch 中，和 ONNX 有关的定义全部放在`torch.onnx`目录中。`symbolic_opset{n}.py`（符号表文件）即表示 PyTorch 在支持第 n 版 ONNX 算子集时新加入的内容。
