---
layout: post
title: deploy_model_on_raspi_with_ncnn
date: 2024-11-03
tags: [raspi]
author: taot
---

## 使用NCNN在树莓派部署深度学习模型流程

### 0 端侧部署深度学习模型流程

一般来说，端侧深度学习模型部署可以分成如下几个阶段：
* 模型训练，利用数据集训练模型，面向端侧设计模型时，需要考虑模型大小和计算量
* 模型压缩，主要优化模型大小，可以通过剪枝、量化等手段降低模型大小
* 模型部署，包括模型管理和部署、运维监控等；
* 端侧推理，加载模型，完成推理相关的所有计算

基于 pytorch 训练模型 --> torch 模型转成 onnx 格式 --> onnx 模型转成 NCNN 的模型格式 --> 在 NCNN 中进行模型推理

#### 0.1 onnx

ONNX:（Open Neural Network Exchange）：开放神经网络交换, 是一种针对机器学习算法所设计的开放式文件格式标准，用于存储训练好的算法模型。许多主流的深度学习框架（如 PyTorch、TensorFlow、MXNet）都支持将模型导出为 ONNX 模型。ONNX 使得不同的深度学习框架可以以一种统一的格式存储模型数据以及进行交互。

ONNX Runtime：ONNX运行时，支持Linux、Windows、Mac OS、Android、iOS等多种平台和多种硬件（CPU、GPU、NPU等）上进行模型部署和推理。

#### 0.2 NCNN

ncnn 是一个为手机端极致优化的高性能神经网络前向计算框架。 ncnn 从设计之初深刻考虑手机端的部署和使用。 无第三方依赖，跨平台，手机端 cpu 的速度快于目前所有已知的开源框架。 基于 ncnn，开发者能够将深度学习算法轻松移植到手机端高效执行。

### 1 torch 模型转 onnx

#### 1.1 torch 模型保存

* 只保存模型参数
```python
# 保存模型参数
torch.save(model.state_dict(), "./data/model_parameter.pth")

# 调用模型Model
new_model = Model()
# 加载模型参数
new_model.load_state_dict(torch.load("./data/model_parameter.pth"))
# 前向推理
new_model.forward(input)
```

* 保存完整模型
```python
# 保存整个模型
torch.save(model, './data/model.pth')
# 加载模型
new_model = torch.load('./data/model.pkl')
```

> 一般仅仅保存模型参数，用的时候创建模型结构，然后加载模型参数。


#### 1.2 torch 模型转 onnx

```python
# onnx模型导入
import onnx
import torch
import torch.nn as nn
import torchvision.models as models
from onnxsim import simplify

# 构建模型结构的 class
class Net(nn.Module):
    def __init__(self, prior_size):
        func
    def forward(self, x):
        func
        return x

# 初始化模型结构并加载模型参数
model = Net(prior_size)
model.load_state_dict(torch.load('xxx.pth', map_location='cpu'))
model.eval()

# torch 模型转 onnx
input_names = ['input']
output_names = ['output']
x = torch.randn(1, 3, 240, 320, requires_grad=True)
# 转onnx接口调用
torch.onnx.export(
    model, 
    x, 
    'model.onnx', 
    input_names=input_names, 
    output_names=output_names, 
    verbose='True'
)
# 优化onnx模型
onnx_model = onnx.load_model('model.onnx', load_external_data=True)
# simply model
model_simp, check = simplify(onnx_model)
onnx.save(
    model_simp, 
    'model.onnx', 
    save_as_external_data=True, 
    all_tensors_to_one_file=True
)
```

#### 1.3 验证 onnx 转换结果

```python
import onnx

# 加载onnx模型
onnx_model = onnx.load("best.onnx")
# 验证onnx模型是否成功导出
onnx.checker.check_model(onnx_model)
# 如果没有报错，表示导出成功
```

#### 1.4 测试 onnx 模型
```python
import onnxruntime
import numpy as np

# 创建会话
#  cuda
# session = onnxruntime.InferenceSession("model.onnx", providers=['CUDAExecutionProvider'])
# cpu
session = onnxruntime.InferenceSession("model.onnx", providers=['CPUExecutionProvider'])
x = np.random.randn(1, 3, 240, 320)
ort_input = {session.get_inputs()[0].name: x.astype(np.float32)}
ort_output = session.run(None, ort_input)
# # 如果没有报错，表示执行成功
```

### 2 onnx 模型转 NCNN 格式

#### 2.1 编译 NCNN

* 项目代码克隆
```bash
git clone git@github.com:Tencent/ncnn.git
```

* 使用 cmake 编译：
```bash
# 在 ncnn 项目目录下
mkdir build && cd build
cmake ..
make -j4
```

* 安装，编译完成后，在编译目录执行 make install 就可以完成到指定目录的安装
```bash
cd build
make install
```

#### 2.2 下载 onnx2ncnn tool

可以自己手动编译出onnx2ncnn，也可以在其官方仓库里直接下载编译好的可执行文件。这个工具依赖 protobuf，编译起来比较麻烦，可以直接去官方仓库里直接下载：https://github.com/Tencent/ncnn/releases，根据操作系统选择合适的版本下载即可，zip 里面就有 onnx2ncnn 文件。

#### 2.3 onnx 转 ncnn

```bash
./ncnn-20240820-ubuntu-2404/bin/onnx2ncnn model.onnx model.param model.bin
```

### 3 模型推理调用

#### 3.1 C++ 调用推理

这里先简单地实现一个加载模型的代码：
```cpp
#include <iostream>
#include <chrono>
#include "net.h"
using namespace std;
 
int main() {    
    const char *binfile = "./model.bin";
    const char *paramfile = "./model.param";
    static ncnn::Net* testNet = NULL;
 ​
    testNet = new ncnn::Net();
    testNet->load_param(paramfile);
    testNet->load_model(binfile);
}
```

编写 CMakeLists.txt:
```bash
cmake_minimum_required(VERSION 3.8)
project(ncnn_test)
set(SOURCE_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/test.cpp
)
 ​
set(ncnn_DIR "<ncnn_install_dir>/lib/cmake/ncnn" CACHE PATH "Directory that contains ncnnConfig.cmake")
find_package(ncnn REQUIRED)

add_executable(${PROJECT_NAME} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} PRIVATE ncnn)
```

这里由于是自己编译的 ncnn，`ncnn_install_dir` 此时对应 `build/install` 目录。

使用 cmake 构建项目：
```bash
mkdir build & cd build
cmake ..
make
```

build 完成，在 build 目录下有一个可执行文件`ncnn_test`。

后续还可以使用 C++ 调用和封装，大概流程是先加载模型创建一个指向模型的指针，然后创建session、创建用于处理输入的tensor，将input_tensor送入session，运行session，最后得到网络的输出。

