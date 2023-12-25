---
layout: post
title: termux_install_setting
date: 2023-12-26
---

## termux 配置和系统安装

### 1、termux  配置

去 [termux官网](https://f-droid.org/packages/com.termux/) 下载对应的 APK 并安装即可

Termux 是 Android 平台上的一个终端模拟器，它将众多 Linux 上运行的软件和工具近乎完美的移植到了手机端。

*   换源

    由于 Termux 默认使用国外的镜像源，在国内访问国外服务器会很慢或者连接不上，因此需要将镜像源换成国内的镜像源

    使用 vi 编辑 $PREFIX/etc/apt/sources.list 修改为如下内容，`vi $PREFIX/etc/apt/sources.list`

    ```bash
    # 使用中科大的镜像源
    deb https://mirrors.ustc.edu.cn/termux/apt/termux-main stable main
    ```

*   配置 ssh

    通过配置 ssh，可以通过电脑远程连接（window 下的各种 terminal 工具或者是 Linux terminal 直接连接），这样可以降低眼睛的耗损，使用电脑也方便代码编辑

    *注：在没有内网穿透的情况下，可以 ssh 连接的前提是手机和电脑处于同一局域网下*

    *   安装 ssh

        ```bash
        pkg install openssh #安装openssh
        
        passwd #安装完成配置密码，输入两次
        pkg install nmap #安装nmap
        
        # 启动ssh
        sshd
        ifconfig	# 查看当前 IP
        nmap [ip地址] #上面sshd与nmap来开启服务
        ```

    *   电脑 ssh 连接

        ```bash
        # 例如在 Windows 下使用 XShell 连接
        
        # 在手机上
        whoami	# 查看当前用户名
        
        # XShell 上
        ssh user_id@IP:8022		# 一般默认端口是 8022
        ```

    *   设置 sshd 自启动

        ```bash
        # 重启termux需要重新输入sshd和nmap命令来开启ssh服务, 否则无法连接，因此需要配置 sshd 自启动
        # 根目录创建.bashrc
        cd
        touch .bashrc
        
        # 编辑自启动脚本
        vim .bashrc
        ```

    *   sshd自启动脚本，其他应用程序同理

        ```bash
        # 初始化sshd
        if pgrep -x "sshd" >/dev/null
          then
            echo " > sshd started"
          else
            sshd >/dev/null
            echo " > sshd start success"
        fi
        ```

    *   设置息屏不断开

        termux息屏后会导致ssh断连，应用程序中断等情况，需要设置常驻后台

        ```bash
        # 命令行开启
        ~ $ termux-wake-lock
        ```

*   安装 Nodejs

    ```bash
    # 安装长期支持版nodejs-lts
    ~ $ pkg install nodejs-lts
    
    # 安装完成后使⽤如下命令查看版本信息
    node --version
    npm --version
    ```

    *   测试 hello world

        ```bash
        # 编辑HelloWorld.js文件
        ~ $ vim HelloWorld.js
        
        # 文件内容
        console.log('Hello World!');
        
        # 运⾏HelloWorld.js
        ~ $ node HelloWorld.js
        Hello World!
        ```

### 2、安装 ubuntu

```bash
pkg install proot wget -y
pkg installl proot-distro
proot-distro list
# 这时会列出 ARM 所支持的所有linux版本

# 这时我们选 ubuntu
# 安装
proot-distro install ubuntu

# 安完之后登录
proot-distro login ubuntu

# 防止以后忘记怎么登录，可以在 termux home 目录新建文件来记录命令
vim start_cmd.txt
# 文件内容
ubuntu：
proot-distro login ubuntu

# 接下来就可以在 termux ubuntu 愉快地玩耍了
```



### 3、安装 kali Linux

```bash
# 安装 git
pkg install git

# 安装 python3 和 python2（安装python2只是为了增加兼容性）
pkg install python
pkg install python2

# clone kali os 相关资源
git clone git@github.com:Taot-chen/kali_install.git
cd kali_install/install_kali/

# 给脚本添加执行权限
chmod +x kali_nethunter touchup.sh

# 安装 kali
./kali_nethunter
# 安装完成然后再输入 startkali 启动 kali linux

#start_cmd.txt 添加 启动命令
kali:
startkali

# Kali Linux is not about its tools, nor the operating system. Kali Linux is a platform.
```

