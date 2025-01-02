---
layout: post
title: configure_Alibaba_cloud_server_as_proxy_server
date: 2025-01-02
tags: [linux]
author: taot
---


## 阿里云服务器配置成代理服务器

### 1 安装Squid

* 安装Squid

```bash
sudo apt install squid -y
```

* 修改配置文件

```bash
sudo vim /etc/squid/squid.conf
# 把http_access deny all改为http_access allow all。下面的3128端口，如果已经被占用了的话，需要改一下。
```

阿里云服务器需要去防火墙里把3128端口打开，不然访问不了。

### 2 启动Squid服务

```bash
sudo systemctl start squid # 开启
```

启动之后，查看系统有没有在监控3128端口，输入命令：

```bash
netstat -ntl
```

其他相关命令：

```bash
systemctl stop squid # 停止
service squid restart # 重启
```

* 关闭防火墙

```bash
sudo systemctl stop firewalld
```


### 3 设置通过代理访问

在需要使用代理的Linux机器上：

```bash
export http_proxy=http://ip:port
export https_proxy=https://ip:port
```

永久配置生效，写进 `~/.bashrc`:
```bash
export http_proxy=http://ip:port
export https_proxy=https://ip:port
```

#### 3.1 配置 git 代理

* 为http协议远程仓库设置代理

```bash
#如果使用的是http代理 
git config --global http.proxy http://ip:port
git config --global https.proxy https://ip:port
#如果使用的是socks5代理
git config --global http.proxy socks5://ip:port
```

取消http协议远程仓库代理:

```bash
git config --global --unset http.proxy
git config --global --unset https.proxy
```

* 为git协议远程仓库设置代理

设置git协议的代理，需要安装connect-proxy。

新建文件`gitproxy.sh`在里面输入：

```bash
#!/bin/bash
#-S 为 socks, -H 为 HTTP
connect -S 代理地址:代理端口 $@
```

给文件增加执行属性，并它拷贝到用户路径中，以便可以被git调用:

```bash
chmod +x gitproxy.sh
sudo cp gitproxy.sh /usr/bin/
```

将gitproxy.sh关联到git中：

```bash
git config --global core.gitproxy gitproxy.sh
```


取消git协议远程仓库代理:

```bash
git config --global --unset core.gitproxy
```


* 为ssh协议远程仓库设置代理

设置ssh协议的代理，需要安装connect-proxy

```bash
sudo apt update
sudo apt-get install connect-proxy
```

访问ssh协议远程仓库，相关代理并不通过git来设置，而是设置本地ssh的相关文件。首先在`~/.ssh/`路径下创建名为config的空文件，然后粘贴下面的内容进去：

```bash
Host github.com
#-S 为 socks, -H 为 HTTP
ProxyCommand connect -S 代理地址:代理端口 %h %p
```

取消ssh协议远程仓库代理:

```bash
git config --global --unset core.gitproxy
```

* 前面的 ssh 协议可能会出现类似下面的报错

```bash
/bin/bash: line 1: exec: nc: not found
kex_exchange_identification: Connection closed by remote host
Connection closed by UNKNOWN port 65535
fatal: Could not read from remote repository.

Please make sure you have the correct access rights
and the repository exists.
```

此时可以尝试**改用 HTTPS 协议，走 443 端口**：
在 `~/.ssh/config` 文件中添加下面的配置即可

```bash
Host github.com
    Hostname ssh.github.com
    Port 443
    User git
```
不出意外的话，此时使用 ssh 协议 clone github 仓库是可以的了。
