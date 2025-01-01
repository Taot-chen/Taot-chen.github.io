---
layout: post
title: raspberry_pi_clash
date: 2025-01-01
tags: [raspberry]
author: taot
---

## 树莓派配置 clash

### 1 安装 clash


* 下载

```bash
wget https://github.com/MetaCubeX/mihomo/releases/download/v1.19.0/mihomo-linux-arm64-v1.19.0.gz
```

* 安装 clash

```bash
gzip -d -f mihomo-linux-arm64-v1.19.0.gz
mv mihomo-linux-arm64-v1.19.0 clash

chmod +x clash

# 查看是否安装成功
clash -v
```

### 2 配置

* 添加 `config.yaml` 与 `country.mmdb` 文件

此时的 clash 是不能正常工作的，因为没有 `config,yaml` 配置文件。

在安装 clash 的目录下运行 clash：

```bash
clash
```

会自动生成配置文件，在`~/.config/mihomo` 目录:

```bash
cd ~/.config/mihomo && ls
```

覆盖原有的 `config,yaml`: 最简单的方式是将成功科学的 Windows clash 客户端配置文件拖入树莓派。也可以在机场找机场订阅地址。

下载 country.mmdb 文件：​ country.mmdb 为全球 IP 库，可以实现各个国家的 IP 信息解析和地理定位。clash 运行会自动下载这个文件，也可以手动下载并放入目录。在[meta-rules-dat](https://github.com/MetaCubeX/meta-rules-dat/releases)的发行包里找到 `country.mmdb` 文件，下载完成后同样放入`~/.config/mihomo` 目录。

* 配置 raspi-config

​ 此时 clash 已经正常运行了，但是树莓派还需要配置网络代理，才能完成科学上网。

```bash
sudo raspi-config

# 找到 6 Advanced Options 回车进入
# 选择 A3 Network Proxy Settings 进入
# 选择 P1 All 填入 http://127.0.0.1:7890
# Ok 确认 Finish 确认然后选择 reboot 重启设备
```

* 进入终端输入 clash 启动

不出意外的话，此时就已经可以使用了，可以`wget www.google.com`测试一下。

### 3 clash 开机自启

* 移动 clash 到`/usr/local/bin/`目录，注意命令中 clash 是树莓派中 clash 的路径位置：

```bash
sudo mv clash /usr/local/bin/
```

* 在`/etc` 目录下创建 clash 目录

```bash
sudo mkdir /etc/clash
```

* 复制 `config.yaml` 与 `Country.mmdb` 文件到`/etc/clash` 目录

```bash
sudo cp -f ~/.config/mihomo/config.yaml ~/.config/mihomo/country.mmdb /etc/clash
```

* 创建 systemd 配置文件，实现开机自启

```bash
sudo vim /etc/systemd/system/clash.service
```

写入以下内容:

```bash
[Unit]
Description=mihomo Daemon, Another Clash Kernel.
After=network.target NetworkManager.service systemd-networkd.service iwd.service

[Service]
Type=simple
LimitNPROC=500
LimitNOFILE=1000000
CapabilityBoundingSet=CAP_NET_ADMIN CAP_NET_RAW CAP_NET_BIND_SERVICE CAP_SYS_TIME
AmbientCapabilities=CAP_NET_ADMIN CAP_NET_RAW CAP_NET_BIND_SERVICE CAP_SYS_TIME
Restart=always
ExecStartPre=/usr/bin/sleep 1s
ExecStart=/usr/local/bin/clash -d /etc/clash
ExecReload=/bin/kill -HUP $MAINPID

[Install]
WantedBy=multi-user.target
```

* 启用 clashclash.service 服务

```bash
sudo systemctl enable clash
```

* 启动 clash

```bash
sudo systemctl start clash
```

* 重新加载 clash

```bash
sudo systemctl reload clash
```

* 查看运行状况

```bash
sudo systemctl status clash
```


* 查看运行日志，如果出错可以在这里捕捉 error 信息

```bash
journalctl -u clash -o cat -f
```

* clash 在线面板

浏览器输入https://d.metacubex.one使用图形化操作界面

