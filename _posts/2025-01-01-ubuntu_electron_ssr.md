---
layout: post
title: ubuntu_electron_ssr
date: 2025-01-01
tags: [ubuntu]
author: taot
---

## ubuntu配置electron_ssr

### 1 安装electron-ssr

下载electron-ssr：[electron-ssr](https://github.com/qingshuisiyuan/electron-ssr-backup/releases/download/v0.2.6/electron-ssr-0.2.6.deb)

* 安装依赖：
```bash
sudo apt install libcanberra-gtk-module libcanberra-gtk3-module gconf2 gconf-service libappindicator1
sudo apt-get install libssl-dev 
sudo apt-get install libsodium-dev
sudo apt-get install libnotify4
```

* 安装electron-ssr
```bash
sudo dpkg -i electron-ssr-0.2.6.deb
```

### 2 运行配置

* 运行

```bash
electron-ssr

# 弹出窗口，点击下载electron-ssr
```

* 配置

可以先试试在订阅管理中，添加订阅；如果无法添加订阅，可以单个添加飞机。

在**系统设置->网络设置->网络代理设置**中，选择自动，其中`configuration URL` 会自动填入`http://127.0.0.1:2333/proxy.pac`，注意，如果这个端口占用了，需要更换其他空闲端口。

**如果在ssr中已经添加了订阅，并且订阅是可用的，但是仍然无法冲浪，极有可能就是代理设置没填好**。

完成上述步骤之后，重启浏览器，google 等网站就可访问了


### 3 终端配置

* 安装privoxy

```bash
sudo apt-get install privoxy 
```

* 在/etc/privoxy/config文件中添加如下的一条配置：

```bash
forward-socks5 / 127.0.0.1:1080 .
```

* 重启provixy服务，并确认已经启动

```bash
service privoxy restart
service privoxy status
```

privoxy服务默认跑在本地的**8118端口**

* 安装polipo

```bash
sudo apt-get install polipo 
```

* 在/etc/polipo/config文件中做如下的配置：

```bash
# This file only needs to list configuration variables that deviate
# from the default values.  See /usr/share/doc/polipo/examples/config.sample
# and "polipo -v" for variables you can tweak and further information.
logSyslog = true
logFile = /var/log/polipo/polipo.log
proxyAddress = "0.0.0.0"
socksParentProxy = "127.0.0.1:1080"
socksProxyType = socks5
chunkHighMark = 50331648
objectHighMark = 16384
dnsQueryIPv6 = no
```

* 重启polipo服务并确认
```bash
service polipo restart
service polipo status
```

polipo服务默认跑在本地的**8123端口**。

* 终端设置

```bash
# 当前终端设置代理
export http_proxy=http://127.0.0.1:8118
export https_proxy='http://127.0.0.1:8118'

# 在.bashrc文件中设置代理
alias proxy="export http_proxy=http://localhost:8118;export https_proxy=http://localhost:8118"
alias unproxy="unset http_proxy;unset https_proxy"
# 生效配置 source .bashrc
```

终端启动代理或者终止代理(以后每次需要终端代理，都要先启动) proxy 或者 unproxy

* 测试终端

```bash
wget www.google.com
```
