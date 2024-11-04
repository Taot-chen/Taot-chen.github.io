---
layout: post
title: raspi_deploy_csi_video_stream_with_mobilenet_v2
date: 2024-11-02
tags: [raspi]
author: taot
---

## 树莓派5使用 MobileNet v2 实时推理相机视频

Pytorch 对于树莓派提供了较好的支持，可以利用 Pytorch 在树莓派上进行试试推理，本文将尝试在树莓派5上使用 CPU 实时推理 MobileNet v2 分类模型。github 项目: https://github.com/Taot-chen/raspberrypi_dl

### 0 准备

* 树莓派 5 或者树莓派 4 Model B 2GB以上内存即可
* 树莓派摄像头模块
* 散热片和风扇（可选，但建议使用）
* 5V 5A USB-C 电源（针对树莓派5）
* SD 卡（至少 8GB）
* SD 卡读卡器/写入器

PyTorch 仅为 Arm 64 位 (aarch64) 提供 pip 包，因此需要在树莓派上安装 64 位版本的 OS。树莓派系统安装过程很平常，这里不再赘述，有需要可以参看这篇博客;https://blog.csdn.net/qq_38342510/article/details/142289792。

### 1 安装 PyTorch 和 OpenCV

PyTorch 和需要的所有其他库都有 ARM 64 位/aarch64 变体，因此可以通过 pip 安装它们，并且像任何其他 Linux 系统一样工作：
```bash
python3 -m pip install torch torchvision torchaudio
python3 -m pip install opencv-python
python3 -m pip install numpy --upgrade
```

检查 torch 是否一切安装正确：
```bash
python3 -c "import torch; print(torch.__version__)"
python3 -c "import cv2; print(cv2.__version__)"
```

### 2 视频采集

对于视频采集，使用 OpenCV 来传输视频帧。 这里不使用 picamera 是因为 picamera 在 64 位 Raspberry Pi OS 上不可用，并且比 OpenCV 慢得多。OpenCV 直接访问 /dev/video0 设备来抓取帧。

这里使用的模型 (MobileNetV2) 接收 224x224 的图像大小，因此可以直接从 OpenCV 以 36fps 的速度请求该大小。目标模型帧率为 30fps，但我们请求的帧率略高于此值，以便始终有足够的帧。

```python
import cv2
from PIL import Image

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 224)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 224)
cap.set(cv2.CAP_PROP_FPS, 36)
```

OpenCV 返回一个 BGR 格式的 numpy 数组，因此需要读取并进行一些调整才能将其转换为预期的 RGB 格式:
```bash
ret, image = cap.read()
# convert opencv output from BGR to RGB
image = image[:, :, [2, 1, 0]]
```

### 3 图像预处理

获取帧并将其转换为模型期望的格式:
```bash
from torchvision import transforms

preprocess = transforms.Compose([
    # convert the frame to a CHW torch tensor for training
    transforms.ToTensor(),
    # normalize the colors to the range that mobilenet_v2/3 expect
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(image)
# The model can handle multiple images simultaneously so we need to add an
# empty dimension for the batch.
# [3, 224, 224] -> [1, 3, 224, 224]
input_batch = input_tensor.unsqueeze(0)
```

### 4 模型加载

aarch64 版本的 pytorch 需要使用 qnnpack 引擎：
```bash
import torch
torch.backends.quantized.engine = 'qnnpack'
```

为了获得更好的推理性能，使用量化和融合的模型。量化意味着使用 int8 进行计算，比标准的 float32 数学运算性能高得多；融合意味着连续的操作已尽可能融合在一起，例如，像激活函数 (ReLU) 这样的东西可以在推理过程中合并到前一层 (Conv2d) 中。使用 torchvision 开箱即用的预量化和融合版本的 MobileNetV2：
```bash
from torchvision import models
net = models.quantization.mobilenet_v2(pretrained=True, quantize=True)
```

通过 jit 模型，以减少 Python 开销并融合任何操作，从而提高推理性能：
```bash
net = torch.jit.script(net)
```
如果禁用了用户界面和默认启用的所有其他后台服务，可以获得更高的性能和更稳定的推理帧率。


完整推理代码:
```python
import time

import torch
import numpy as np
from torchvision import models, transforms

import cv2
from PIL import Image

torch.backends.quantized.engine = 'qnnpack'

cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 224)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 224)
cap.set(cv2.CAP_PROP_FPS, 36)

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

net = models.quantization.mobilenet_v2(pretrained=True, quantize=True)
net = torch.jit.script(net)

started = time.time()
last_logged = time.time()
frame_count = 0

with torch.no_grad():
    while True:
        # read frame
        ret, image = cap.read()
        if not ret:
            raise RuntimeError("failed to read frame")

        # convert opencv output from BGR to RGB
        image = image[:, :, [2, 1, 0]]
        permuted = image

        # preprocess
        input_tensor = preprocess(image)

        # create a mini-batch as expected by the model
        input_batch = input_tensor.unsqueeze(0)

        # run model
        output = net(input_batch)
        # do something with output ...

        # log model performance
        frame_count += 1
        now = time.time()
        if now - last_logged > 1:
            print(f"{frame_count / (now-last_logged)} fps")
            last_logged = now
            frame_count = 0
```


