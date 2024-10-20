---
layout: post
title: raspberry_pi_gpu
date: 2024-10-20
tags: [RaspberryPi]
author: taot
---

## 树莓派5 gpu 加速

树莓派 5 配备了时钟频率 800 MHz VideoCore VII GPU，是用于图形处理的专用硬件，主要用于图形加速（如视频解码、显示等）。这个 GPU 的主要目标不是像 NVIDIA 或 AMD GPU 那样用于高性能通用计算（如深度学习加速）。不过，这个 GPU 支持 OpenGL 和 OpenCL，通过适当的设置和库，还是可以利用 GPU 的部分功能来加速特定的任务。利用 OpenCL 和 TensorFlow Lite 等库，仍然可以在树莓派上进行部分 GPU 加速任务。

树莓派的 GPU 性能虽然比不上专用的高性能 GPU，但在适当的优化下仍然能够胜任一些轻量级的深度学习和计算任务。

### 1 GPU 检查

#### 1.1 查看和验证树莓派 GPU

查看树莓派的硬件配置，确认 GPU 的存在及其参数：
```bash
vcgencmd version

# 2024/09/10 14:40:30
# Copyright (c) 2012 Broadcom
# version 5be4f304 (release) (embedded)
```

通过上面的命令，可以查看当前的 GPU 固件版本等信息，包括 GPU 的版本、驱动程序信息等。

#### 1.2 安装和使用 vcgencmd 工具

`vcgencmd` 是树莓派上一个用来获取 GPU 和 VideoCore 的详细信息的工具。可以通过以下命令安装：
```bash
sudo apt update
sudo apt install raspberrypi-ui-mods
```

#### 1.3 检查 OpenGL 支持

安装glxinfo，可以通过以下命令安装：
```bash
sudo apt install mesa-utils
```

用以下命令检查树莓派是否支持 OpenGL（用于 3D 图形和 GPU 计算）：
```bash
glxinfo | grep "OpenGL"

OpenGL vendor string: Mesa
OpenGL renderer string: llvmpipe (LLVM 15.0.6, 128 bits)
OpenGL core profile version string: 4.5 (Core Profile) Mesa 23.2.1-1~bpo12+rpt3
OpenGL core profile shading language version string: 4.50
OpenGL core profile context flags: (none)
OpenGL core profile profile mask: core profile
OpenGL core profile extensions:
OpenGL version string: 4.5 (Compatibility Profile) Mesa 23.2.1-1~bpo12+rpt3
OpenGL shading language version string: 4.50
OpenGL context flags: (none)
OpenGL profile mask: compatibility profile
OpenGL extensions:
OpenGL ES profile version string: OpenGL ES 3.2 Mesa 23.2.1-1~bpo12+rpt3
OpenGL ES profile shading language version string: OpenGL ES GLSL ES 3.20
OpenGL ES profile extensions:
```
上面输出了 OpenGL 的版本、供应商信息以及渲染器信息。

#### 1.4 检查 Vulkan 支持

Vulkan 是一种现代的图形 API，通常可以更好地访问 GPU 的底层功能。

安装 vulkaninfo，使用以下命令来安装：
```bash
sudo apt install vulkan-tools
```
```bash
vulkaninfo
```
输出 GPU 的供应商和型号（通常是 Broadcom 的 VideoCore），则表示树莓派5可能支持 Vulkan。


#### 1.5 检查 PyTorch 和 TensorFlow 的 GPU 支持

要在树莓派5上使用 PyTorch 或 TensorFlow 等深度学习框架的 GPU 加速，需要先检查它们是否可以使用 GPU：

* PyTorch：树莓派目前通常使用 PyTorch 的 CPU 版本。在树莓派5上使用 PyTorch 的 GPU 加速相对复杂，因为 PyTorch 通常与 CUDA（NVIDIA 的 GPU 计算平台）绑定。VideoCore GPU 不支持 CUDA，因此需要使用特定的 OpenCL 计算库。
* TensorFlow Lite GPU Delegate：TensorFlow Lite 提供 GPU Delegate，可以利用 OpenCL 进行部分计算加速。要检查 TensorFlow Lite 是否能使用 GPU Delegate，可以安装 TensorFlow Lite 并测试。

安装 TensorFlow Lite，由于 tflite 对于 aarch64 的支持最高只到 python3.8，因此需要可以先创建 python3.8 虚拟环境：
```bash
conda create -n py38 python==3.8.11
wget wget https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp38-cp38-linux_aarch64.whl
python3 -m pip install tflite_runtime-2.1.0.post1-cp38-cp38-linux_aarch64.whl

# 测试安装是否成功
python3
import tflite_runtime.interpreter as tflite
# 没有报错即是成功

# 安装 tensorflow
sudo apt update
sudo apt install libatlas-base-dev
python3 -m pip install keras_applications==1.0.8 --no-deps
python3 -m pip install keras_preprocessing==1.1.0 --no-deps
sudo apt-get install libhdf5-dev
python3 -m pip install h5py==2.9.0
python3 -m pip install -U six wheel mock
python3 -m pip install tensorflow
# 验证 TensorFlow
python -c "import tensorflow as tf;print(tf.reduce_sum(tf.random.normal([1000, 1000])))" 
# 输出 tf.Tensor(1035.3678,shape=(),dtype=float32)则表示安装成功
```

测试 GPU 支持：
```bash
import tensorflow as tf

# 检查 TensorFlow 版本
print("TensorFlow version:", tf.__version__)

# 检查 GPU 支持
try:
    from tensorflow.lite.experimental.delegate import load_delegate
    print("GPU support is enabled.")
except ImportError:
    print("GPU support is not available.")
```

### 2 安装并配置 GPU 加速库

#### 2.1 安装 OpenCL

使用 GPU 加速来加速深度学习或图像处理任务，可以安装 OpenCL 库，树莓派5可以安装 OpenCL 库，如pyopencl，来访问 VideoCore GPU 进行通用计算。
```bash
sudo apt install ocl-icd-libopencl1
sudo apt install clinfo
```
运行`clinfo`检查 OpenCL 平台是否可用。
```bash
clinfo

Number of platforms                               0

ICD loader properties
  ICD loader Name                                 OpenCL ICD Loader
  ICD loader Vendor                               OCL Icd free software
  ICD loader Version                              2.3.1
  ICD loader Profile                              OpenCL 3.0
```

#### 2.2 使用 glmark2 测试 GPU 性能

```bash
sudo apt install glmark2
glmark2

# =======================================================
#                                   glmark2 Score: 20
# =======================================================
```

### 3 使用 GPU 加速 opencv

在树莓派上使用 GPU 加速 opencv 需要满足：
* 树莓派使用待遇 GPU 支持的操作系统版本，树莓派官方的 Raspberry Pi OS 的最新版本支持
* 需要使用 OpenCV 的 DNN 模块和 OpenCL

#### 3.1 安装 OpenCL
```bash
sudo apt-get update
sudo apt-get upgrade
sudo apt-get install ocl-icd-opencl-dev
```

#### 3.2 安装 OpenCV
```bash
sudo apt-get update
sudo apt-get upgrade
sudo apt-get install python3-opencv
```

#### 3.3 使用 OpenCV 的 DNN 模块，在 Python 中使用以下代码：
```python
import cv2
net = cv2.dnn.readNetFromTensorflow(model_path, config_path)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL)
```
注：
* 这里的 model_path 和 config_path 是模型的路径和配置文件路径
* DNN_BACKEND_OPENCV 表示使用 OpenCV 的 DNN 模块
* DNN_TARGET_OPENCL 表示使用 OpenCL 库及进行加速
* 这里可以使用 TensorFlow 模型，也可以使用 Caffe、Darknet、ONNX
