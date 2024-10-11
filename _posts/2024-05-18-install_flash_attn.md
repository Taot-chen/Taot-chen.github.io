---
layout: post
title: install_flash_attn
date: 2024-05-18
tags: [git]
author: taot
---

## ubuntu安装flash_attn

安装 flash_attn 需要注意：
    * flash_sttn 依赖 cuda-11.6及以上的版本，使用命令 `nvcc --version` 查看 cuda 的版本，[cuda下载地址](https://link.zhihu.com/?target=https%3A//developer.nvidia.com/cuda-toolkit-archive)
    * 检查 pytorch 版本和 cuda 版本是否匹配
    * 需要提前安装`ninja`，否则编译过程会持续很长时间，如果`ninja`已经安装完毕，可以直接执行`pip install flash-attn --no-build-isolation` 来安装 flash_attn
    * 即便是提前安装好了`ninja`，直接`pip`的话编译过程还是会超级慢，可以使用源码安装：
    ```bash
    git clone https://github.com/Dao-AILab/flash-attention.git
    cd flash-attention
    python3 setup.py install
    ```
    * 另外，目前原生的`flash attention`仅支持`Ampere`、`Hopper`等架构的GPU，例如：A100、H100等，V100属于`Volta`架构并不支持。
