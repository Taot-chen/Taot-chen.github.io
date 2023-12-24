---
layout: post
title: ai_inference_accelerate
date: 2023-12-24
---

## AI 图像推理加速

AI 推理加速器

1、CNN 原理

卷积核 --> Relu --> 池化 --> 全连接 --> softmax

两个方向：针对训练过程、针对推理过程

面向推理过程的加速器

*   卷积：乘法器，加法器
*   Relu：激活函数
*   池化：图像缩小，去掉无用信息，压缩
*   全连接：后面都需要跟 relu
*   softmax：分类，实际是计算概率

2、加速器架构

专用软件 || 专用电路

CPU AI 模型加速推理

