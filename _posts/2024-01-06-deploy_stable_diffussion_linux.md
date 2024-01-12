---
layout: post
title: deploy_stable_diffusion_linux
date: 2024-01-06
tags: [AI]
author: taot
---

## 在 Linux 安装 stable diffusion

由于工作站安装的是 ubuntu，卡也在上面，就只能在 ubuntu 上部署安装 stable diffusion 了。另外，Linux 上使用 stable diffusion 也会方便很多。

### 1 准备工作

* NVIDIA 官网下载驱动，主要是为了规避多卡驱动不同的问题。由于本机是两张一样的卡，就可以省去这一步。如果存在不同型号的多卡驱动不兼容的问题，就需要去官网下载。
* 安装 python 3.10
* 安装 CUDA11.8（pytorch2.x，xformers），对 stable diffusion 兼容比较好
    * 支持 pytorch2.x
    * 支持 xformers，可以加速图片生成
* github stable diffusion webUI
    ```bash
        git clone https://github.com/AUTOMATIC1111/stable-diffusion-webui.git
    ```
* 配置 venv python 虚拟环境，因为不同模型的 python 版本要求不同
    ```bash
        # 创建虚拟环境
        python3 -m venv venv

        # 进入虚拟环境
        source venv/bin/activate
        # 退出虚拟环境
        deactivate

    ```
    也可以使用 conda 来进行虚拟环境的创建和管理。

* Stable diffusion WebUI 启动，自动安装依赖
    ```bash
        # 启动，会自动下载依赖
        ./webui.sh --xformers
    ```

### 2 报错解决
'''
这里可能会出现一些报错
1. Cannot locate TCMalloc（improves CPU memory usage），这个报错是因为缺少 libgoogle-perftools4 和 libtcmalloc-minimal4，直接安装即可
sudo apt install libgoogle-perftools4 libtcmalloc-minimal4 -y

1. This scripts must not be launched as root, aborting...
解决方法：
bash webui.sh -f
'''

