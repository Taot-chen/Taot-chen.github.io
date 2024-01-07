---
layout: post
title: personal_deepLearning_workstation_setup
date: 2023-12-27
tags: [AI]
author: taot
---

## 搭建个人深度学习工作站
### 1 硬件平台

### 2 系统安装
作为开发机器用来自己捯饬，使用的是 Ubuntu 系统

* 使用的是 ultralOS 来制作 Ubuntu 的系统启动盘
  * [ultralOS 下载](https://www.cn.ultraiso.net/xiazai.html)
    下载试用版即可。安装完成后，打开 ultralOS，选择继续试用。
    当然，如果有能力，也支持购买完全版。
  * ubuntu18.04 下载：[ 清华大学开源软件镜像站-ubuntu18.04](https://mirrors.tuna.tsinghua.edu.cn/ubuntu-releases/18.04/)
  * 使用 ultralOS 打开刚才下载的 iso 文件
  * 制作启动盘
    * $\boxed{启动} \rightarrow \boxed{写入硬盘映像} \rightarrow \boxed{写入方式：USB-HDD+或者USB-HDD} \rightarrow \boxed{便捷启动} \rightarrow \boxed{写入新的驱动器引导扇区} \rightarrow \boxed{syslinux} \rightarrow \boxed{写入}$
    * 后续等待写入完成即可

* 安装系统，基本就是按部就班，网络上各种教程非常多，就不过多赘述了。

### 3 系统环境配置

* 更换镜像源
  ```bash
    # 备份镜像源列表
    cp /etc/apt/sources.list /etc/apt/sources.list.bk
    # 编辑镜像源列表文件，这里替换的是中科大的镜像源
    #  中科大源
    deb https://mirrors.ustc.edu.cn/ubuntu/ bionic main restricted universe multiverse
    deb-src https://mirrors.ustc.edu.cn/ubuntu/ bionic main restricted universe multiverse
    deb https://mirrors.ustc.edu.cn/ubuntu/ bionic-updates main restricted universe multiverse
    deb-src https://mirrors.ustc.edu.cn/ubuntu/ bionic-updates main restricted universe multiverse
    deb https://mirrors.ustc.edu.cn/ubuntu/ bionic-backports main restricted universe multiverse
    deb-src https://mirrors.ustc.edu.cn/ubuntu/ bionic-backports main restricted universe multiverse
    deb https://mirrors.ustc.edu.cn/ubuntu/ bionic-security main restricted universe multiverse
    deb-src https://mirrors.ustc.edu.cn/ubuntu/ bionic-security main restricted universe multiverse
    deb https://mirrors.ustc.edu.cn/ubuntu/ bionic-proposed main restricted universe multiverse
    deb-src https://mirrors.ustc.edu.cn/ubuntu/ bionic-proposed main restricted universe multiverse

    # 国内的其他源
    #  阿里源
    deb http://mirrors.aliyun.com/ubuntu/ bionic main restricted universe multiverse
    deb http://mirrors.aliyun.com/ubuntu/ bionic-security main restricted universe multiverse
    deb http://mirrors.aliyun.com/ubuntu/ bionic-updates main restricted universe multiverse
    deb http://mirrors.aliyun.com/ubuntu/ bionic-proposed main restricted universe multiverse
    deb http://mirrors.aliyun.com/ubuntu/ bionic-backports main restricted universe multiverse
    deb-src http://mirrors.aliyun.com/ubuntu/ bionic main restricted universe multiverse
    deb-src http://mirrors.aliyun.com/ubuntu/ bionic-security main restricted universe multiverse
    deb-src http://mirrors.aliyun.com/ubuntu/ bionic-updates main restricted universe multiverse
    deb-src http://mirrors.aliyun.com/ubuntu/ bionic-proposed main restricted universe multiverse
    deb-src http://mirrors.aliyun.com/ubuntu/ bionic-backports main restricted universe multiverse

     # 清华源
     deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ bionic main restricted universe multiverse
     deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ bionic main restricted universe multiverse
     deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ bionic-updates main restricted universe multiverse
     deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ bionic-updates main restricted universe multiverse
     deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ bionic-backports main restricted universe multiverse
     deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ bionic-backports main restricted universe multiverse
     deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ bionic-security main restricted universe multiverse
     deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ bionic-security main restricted universe multiverse
     deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ bionic-proposed main restricted universe multiverse
     deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ bionic-proposed main restricted universe multiverse
  ```

* 必要软件安装
  ```bash
    # 更新软件列表
    sudo apt-get update
    sudo apt-get upgrade

    # install software
    sudo apt-get install vim
    sudo apt-get install ssh
    sudo apt install net-tools
    sudo apt install git
  ```

* 配置 ssh
  这里没有配置远程桌面，主要是两方面的考虑：
  * 一个是自己平时开发基本用不上，使用 ssh 远程连接上去有个 terminal 就基本够用了；
  * 另一方面是我的工作站主机就放在桌底墙角，主显示器很容易就可以切换到工作站的视频信号输出，因此没必要远程桌面。
  ```bash
    # 查看本机 IP
    ifconfig

    # 在 ~/.bashrc 中添加sshd自启动脚本，其他应用程序同理
    # 下面的代码添加在 ~/.bashrc 末尾
    # 初始化sshd
    if pgrep -x "sshd" >/dev/null
      then
        echo " > sshd started"
      else
        sshd >/dev/null
        echo " > sshd start success"
    fi
  ```

* 安装 python 和 pip
  ```bash
    sudo apt install python3
    sudo apt install python3-pip
  ```
  如果需要指定版本的 python，可以在后面安装完 conda 环境之后再安装，也方便管理
  * 替换 pip 源
    ```bash
      sudo vim ~/.pip/pip.conf
      # 把文件内容修改为如下内容（清华源）
      [global]
      index-url = https://pypi.tuna.tsinghua.edu.cn/simple/ 
      [install]
      trusted-host = pypi.tuna.tsinghua.edu.cn
    ```
  * 更改默认python版本
    我的习惯是 python 链接到 python2，python3 链接到 python3，python3-pip 链接到 python-pip3
    ```bash
      # 删除原来的python软链接
      sudo rm /usr/bin/python
      sudo rm /usr/bin/python3

      # 新建软链接
      sudo ln -s /usr/bin/python2 /usr/bin/python
      sudo ln -s /usr/bin/pip3 /usr/bin/pip
      sudo ln -s /usr/bin/python3 /usr/bin/python3
    ```

### 4 DeepLearning 开发环境配置

* 安装 NVIDIA 显卡驱动
  
  在图形化界面安装比较方便
  $\boxed{software&update} \rightarrow \boxed{additional drivers}$
  
  之后选择 NVIDIA 的最近新的驱动 apply 即可，稍等几分钟就可以安装完成
  安装完成之后，更新软件列表
  ```bash
    sudo apt update
    sudo apt upgrade
  ```
  此时运行 `nvidia-smi` 可能会报错，重启机器即可

* 安装 cuda
  如果之前安装了旧版本的cuda和cudnn的话，需要先卸载后再安装：
  ```bash
     sudo apt-get remove --purge nvidia*
  ```
  然后按照前面的方法重新安装显卡驱动，安装好了之后开始安装CUDA，如果没有安装过 cuda，可以不卸载
  * 在 NVIDIA 官网下载 cuda 安装包：https://developer.nvidia.com/cuda-toolkit-archive。网站进去可能会很慢，下载速度还是很快的
    * 我这里用的显卡是 2080ti，因此选择版本是：linux-x86_64-ubuntu-18.04-runfile(local)
  * 运行下面的命令进行安装:
    ```bash
      chmod +x cuda_10.1.105_418.39_linux.run
      sudo sh ./cuda_10.1.105_418.39_linux.run
    ```
    安装过程中需要选择要安装选项，不要勾选第一个安装显卡驱动的，因为之前已经安装过了。
  * 设置环境变量
    ```bash
      vim ~/.bashrc
      # 在文件末尾添加
      export CUDA_HOME=/usr/local/cuda-10.1/
      export LD_LIBRARY_PATH=${CUDA_HOME}/lib64
      export PATH=${CUDA_HOME}/bin:${PATH}
    ```
    使环境变量生效
    ```bash
      source ~/.bashrc
    ```
  * 查看安装版本信息`nvcc -V`
    * 也可以编译 sample 程序来验证是否成功
    ```bash
      cd NVIDIA_CUDA-10.1_Samples/1_Utilities/deviceQuery
      make
      ./deviceQuery
    ```
    若安装成功，会打印出显卡信息

* 安装 CuDNN
  在 CUDNN 官网下载 CUDNN：https://developer.nvidia.com/rdp/cudnn-download
  * 选择和之前cuda版本对应的cudnn版本下载，下载之后是一个 tgz 压缩文件，解压该文件：
  ```bash
    tar -xzvf cudnn-10.1-linux-x64-v8.0.5.39.tgz
  ```
  * 复制相应文件到 cuda 目录：
    ```bash
      sudo cp cuda/lib64/* /usr/local/cuda-10.1/lib64
      sudo cp cuda/include/* /usr/local/cuda-10.1/include/
    ```
  * 拷贝完成之后，可以使用以下命令查看CUDNN的版本信息
    ```bash
      cat /usr/local/cuda/include/cudnn_version.h | grep CUDNN_MAJOR -A 2
    ```

* 安装 conda 环境
  不同的训练框架和版本可能会需要不同的python版本相对应，而且有的包比如numpy也对版本有要求。频繁切换 python 和一些包的版本，容易造成包版本冲突的问题，conda 可以给每个配置建立一个虚拟的python环境，在需要的时候可以随时切换，而不需要的时候也能删除不浪费磁盘资源。
  * conda 官网下载 conda 的 linux 安装包：https://www.anaconda.com/download
  * 安装：
    ```bash
      chmod +x Anaconda3-2023.09-0-Linux-x86_64.sh
      ./Anaconda3-2023.09-0-Linux-x86_64.sh
    ```
    后续就一路 enter 安装下去即可。最后会问是否要初始化conda，输入yes确认，重开终端窗口之后，就可以看到conda环境可用了（base代表默认环境）
  * conda 创建新环境
    ```bash
       conda create --name python_38 python=3.8
    ```
  * 进入指定的 conda 环境：
    ```bash
       conda activate python_38
    ```
  * 退出当前 conda 环境：
    ```bash
      conda deactivate
    ```

* 安装 nvidia-docker
  