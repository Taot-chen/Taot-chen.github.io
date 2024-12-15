---
layout: post
title: linux_tips
date: 2024-12-14
tags: [linux]
author: taot
---


## ubuntu 常见问题解决


### 1 ubuntu server 22.04 设置静态 IP

ubuntu server 在重启之后，IP 会变化，这很不利于作内网穿透，因此很有必要使用静态 IP。

登录进系统之后：
```bash
cd /etc/netplan/
ls
# 我这里是 50-cloud-init.yaml

ifconfig
# 查看网卡接口，我这里是 enp7s0
``` 

编辑 netplan 配置文件`50-cloud-init.yaml`并添加以下内容：
```bash
network:
  renderer: networkd
  ethernets:
    enp7s0:
      addresses:
        - 192.168.5.18/24
      nameservers:
        addresses: [255.255.255.0, 0.0.0.0]
      routes:
        - to: default
          via: 192.168.5.1
  version: 2
```

另外，注意，这个文件开头有下面的注释内容：
```bash
# This file is generated from information provided by the datasource.  Changes
# to it will not persist across an instance reboot.  To disable cloud-init's
# network configuration capabilities, write a file
# /etc/cloud/cloud.cfg.d/99-disable-network-config.cfg with the following:
```
这里写的很明白，因此需要在文件`/etc/cloud/cloud.cfg.d/99-disable-network-config.cfg`中添加以下内容：
```bash
network: {config: disabled}
```

使用以下 netplan 命令应用这些更改：
```bash
sudo netplan apply
```



### 2 ubuntu server 安装 nvidia 驱动

* 换源：
```bash
# 备份原来的源文件
sudo cp /etc/apt/sources.list  /etc/apt/sources.list.bak

# 修改源文件
sudo vim /etc/apt/sources.list
# 默认注释了源码仓库，如有需要可自行取消注释
deb https://mirrors.ustc.edu.cn/ubuntu/ jammy main restricted universe multiverse
# deb-src https://mirrors.ustc.edu.cn/ubuntu/ jammy main restricted universe multiverse
 
deb https://mirrors.ustc.edu.cn/ubuntu/ jammy-security main restricted universe multiverse
# deb-src https://mirrors.ustc.edu.cn/ubuntu/ jammy-security main restricted universe multiverse
 
deb https://mirrors.ustc.edu.cn/ubuntu/ jammy-updates main restricted universe multiverse
# deb-src https://mirrors.ustc.edu.cn/ubuntu/ jammy-updates main restricted universe multiverse
 
deb https://mirrors.ustc.edu.cn/ubuntu/ jammy-backports main restricted universe multiverse
# deb-src https://mirrors.ustc.edu.cn/ubuntu/ jammy-back
```


* 更新软件包:

```bash
sudo add-apt-repository ppa:graphics-drivers/ppa  # 加入官方ppa源
sudo apt update  # 检查软件包更新列表
apt list --upgradable  # 查看可更新的软件包列表
sudo apt upgrade  # 更新所有可更新的软件包
```

* 安装显卡驱动

```bash
ubuntu-drivers devices    # ubuntu检测n卡的可选驱动
sudo apt install nvidia-driver-535-server-open  # 根据自己的n卡可选驱动下载显卡驱动
```

等待驱动安装完成，`nvidia-smi`就可用了。




### 3 修复`Temporary failure in name resolution`

修改 `/etc/resolv.conf`文件

将nameservier 修改为`8.8.8.8`:
```bash
nameserver 8.8.8.8
```

重启系统，`/etc/resolv.conf`文件可能被还原，要确保 DNS 配置持久性，可以按照以下步骤操作：
```bash
# 确认 systemd-resolved 服务正在运行
systemctl status systemd-resolved

# 配置 systemd-resolved
# 编辑 /etc/systemd/resolved.conf 文件
sudo vim /etc/systemd/resolved.conf
# 在 [Resolve] 部分添加 DNS 服务器地址
[Resolve]
DNS=8.8.8.8

# 保存文件并重启 systemd-resolved 服务
sudo systemctl restart systemd-resolved
# 创建一个符号链接 /etc/resolv.conf 指向 systemd 生成的文件(非必要)：
sudo ln -sf /run/systemd/resolve/stub-resolv.conf /etc/resolv.conf
```

**终极解决方案**

前面的方案有些情况不一定有效，可以使用下面的方法：
```bash
sudo apt-get install resolvconf
cd /etc/resolvconf/resolv.conf.d/
sudo vim base
# 添加
nameserver 114.114.114.114
nameserver 114.114.114.115
```

更新配置`sudo resolvconf -u`


### 3 libcusparse.so.12: undefined symbol: __nvJitLinkAddData_12_1, version libnvJitLink.so.12

安装最新的 torch-2.5.1 之后，`import torch`遇到报错：
```bash
libcusparse.so.12: undefined symbol: __nvJitLinkAddData_12_1, version libnvJitLink.so.12
```
有帖子说[pytorch依赖没有正确链接](https://blog.csdn.net/qq_42730750/article/details/139582293)导致，需要重新正确连接；

也有帖子通过[升级 CUDA 来解决](https://blog.csdn.net/yourlei/article/details/135224435);

由于我对于 torch 的版本目前没有使用最新版的需求，因此通过回退 torch 的版本也可以解决。通过安装 2.3.0 版本可以正常使用。

