---
layout: post
title: termux_ubuntu_setting
date: 2023-12-26
tags: [termux]
author: taot
---

## termux_ubuntu 系统配置
### 1、添加新用户

由于 termux/ubuntu 默认创建和登录的是 root 用户，root 用户的 home 目录在 `/root` 下，这个使用起来很不方便，使用 root 用户也不太习惯，有必要创建新用户

```bash
# 创建新用户
useradd -d /home/new_user_name -m new_user_nem	# 新用户 home 目录是 /home/new_user_name，新用户用户名是 -m 后的 new_user_name

# 切户到新建的用户
su new_user_name
cd	# 进入 new_user_name 的 home 目录
# 设置新用户密码
passwd new_user_name
whoami	# 查看当前用户
```

### 2、安装图形桌面

*桌面环境没什么额外的用途，只是徒增资源消耗，就不装了。大部分的开发需求都可以通过在电脑安装的 vscode 使用 ssh 远程连接满足*

### 3、环境配置

*   切换 shell 为 bash

    termux 默认使用 sh shell，使用起来会很不习惯，可以切换为习惯的 shell

    ```bash
    # 查看当前 shell
    echo $SHELL	# 通用
    echo $0		# 部分 shell 不支持
    
    # 查看系统中安装了哪些shell
    cat /etc/shells
    # termux ubuntu 一般默认有 /bin/sh /bin/bash /bin/rbash /bin/dash 等 shell
    
    # 通过修改 /etc/passwd文件来修改用户的默认 shell
    vim /etc/passwd
    # 把文件中对应用户的那一行最后的 shell 修改为目标 shell
    # 重新登录用户即可
    ```

    

*   更换镜像源（在 termux ubuntu 中不需要在更换源，更换后可能会导致装软件的时候无法定位到对应的包）

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
    
    #deb cdrom:[Ubuntu 20.04.3 LTS _Focal Fossa_ - Release amd64 (20210819)]/ focal main restricted
    
    # See http://help.ubuntu.com/community/UpgradeNotes for how to upgrade to
    # newer versions of the distribution.
    deb http://cn.archive.ubuntu.com/ubuntu/ focal main restricted
    # deb-src http://cn.archive.ubuntu.com/ubuntu/ focal main restricted
    
    ## Major bug fix updates produced after the final release of the
    ## distribution.
    deb http://cn.archive.ubuntu.com/ubuntu/ focal-updates main restricted
    # deb-src http://cn.archive.ubuntu.com/ubuntu/ focal-updates main restricted
    
    ## N.B. software from this repository is ENTIRELY UNSUPPORTED by the Ubuntu
    ## team. Also, please note that software in universe WILL NOT receive any
    ## review or updates from the Ubuntu security team.
    deb http://cn.archive.ubuntu.com/ubuntu/ focal universe
    # deb-src http://cn.archive.ubuntu.com/ubuntu/ focal universe
    deb http://cn.archive.ubuntu.com/ubuntu/ focal-updates universe
    # deb-src http://cn.archive.ubuntu.com/ubuntu/ focal-updates universe
    
    ## N.B. software from this repository is ENTIRELY UNSUPPORTED by the Ubuntu 
    ## team, and may not be under a free licence. Please satisfy yourself as to 
    ## your rights to use the software. Also, please note that software in 
    ## multiverse WILL NOT receive any review or updates from the Ubuntu
    ## security team.
    deb http://cn.archive.ubuntu.com/ubuntu/ focal multiverse
    # deb-src http://cn.archive.ubuntu.com/ubuntu/ focal multiverse
    deb http://cn.archive.ubuntu.com/ubuntu/ focal-updates multiverse
    # deb-src http://cn.archive.ubuntu.com/ubuntu/ focal-updates multiverse
    
    ## N.B. software from this repository may not have been tested as
    ## extensively as that contained in the main release, although it includes
    ## newer versions of some applications which may provide useful features.
    ## Also, please note that software in backports WILL NOT receive any review
    ## or updates from the Ubuntu security team.
    deb http://cn.archive.ubuntu.com/ubuntu/ focal-backports main restricted universe multiverse
    # deb-src http://cn.archive.ubuntu.com/ubuntu/ focal-backports main restricted universe multiverse
    
    ## Uncomment the following two lines to add software from Canonical's
    ## 'partner' repository.
    ## This software is not part of Ubuntu, but is offered by Canonical and the
    ## respective vendors as a service to Ubuntu users.
    # deb http://archive.canonical.com/ubuntu focal partner
    # deb-src http://archive.canonical.com/ubuntu focal partner
    
    deb http://security.ubuntu.com/ubuntu focal-security main restricted
    # deb-src http://security.ubuntu.com/ubuntu focal-security main restricted
    deb http://security.ubuntu.com/ubuntu focal-security universe
    # deb-src http://security.ubuntu.com/ubuntu focal-security universe
    deb http://security.ubuntu.com/ubuntu focal-security multiverse
    # deb-src http://security.ubuntu.com/ubuntu focal-security multiverse
    
    # This system was installed using small removable media
    # (e.g. netinst, live or single CD). The matching "deb cdrom"
    # entries were disabled at the end of the installation process.
    # For information about how to configure apt package sources,
    # see the sources.list(5) manual.
    
    ```

*   a wrong operation

    当使用新添加的用户使用 apt 安装软件的时候，可能会出现报错

    这个可能和 termux 对多用户支持不好有关，因此还是不要使用新用户了，避免掉进其他的坑，直接使用 root 吧

    使用 root 安装软件则没有这个问题

*   更新 pip 源

    ```bash
    # 注意使用 root 用户
    cd
    mkdir .pip
    vim ~/.pip/pip.conf
    # 文件中添加内容
    [global]
    index-url = http://pypi.douban.com/simple
    [install]
    trusted-host=pypi.douban.com
    
    ```

*   常用工具安装

    *   `sudo apt install net-tools`

    *   `sudo apt install gcc`

    *   `sudo apt install g++`

    *   `sudo apt install cmake`

    *   apt install pipx

        安装这个包的原因主要是在使用 pip 安装 python 包的时候，可能会有报错

        

        安装完 pipx 之后，将其添加到 PATH`pipx ensurepath`

        

        接下来就可以使用 `pipx install package_nme`来安装 python 包

        另外，也可以强行安装

        ```bash
        pip install --upgrade  --force-reinstall package_name --break-system-packages
        ```

        