---
layout: post
title: model_deployment_3_torch2onnx
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

### 2 `torch.onnx.export` 参数注解

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
前三个必选参数分别为 torch 模型、模型输入（转 ONNX 的时候的 dummy input）、ONNX 模型的保存路径
