---
layout: post
title: ubuntu_rc_local
date: 2024-10-08
tags: [linux]
author: taot
---

## ubuntu下没有rc.local解决方法

新的 Linux 发行版已经没有 rc.local 文件，将其服务化了，这会导致我们如果能只是在 `/etc/rc.local` 中添加开机自启任务，会导致任务自启失败。这里记录了解决方法。

### 1 配置 rc-local.service 服务

* 添加 rc-local.service 服务
```bash
sudo vim /etc/systemd/system/rc-local.service
```

写入如下内容：
```bash
[Unit]
 Description=/etc/rc.local Compatibility
 ConditionPathExists=/etc/rc.local

[Service]
 Type=forking
 ExecStart=/etc/rc.local start
 TimeoutSec=0
 StandardOutput=tty
 RemainAfterExit=yes
 SysVStartPriority=99

[Install]
 WantedBy=multi-user.target
```

* 激活 rc-local.service 服务
```bash
sudo systemctl enable rc-local.service
```

### 2 创建`/etc/rc.local`文件
```bash
sudo vim /etc/rc.local
```

写入如下内容：
```bash
#!/bin/sh -e
# 
# rc.local
#
# This script is executed at the end of each multiuser runlevel.
# Make sure that the script will "exit 0" on success or any other
# value on error.
#
# In order to enable or disable this script just change the execution
# bits.
#
# By default this script does nothing.

# 开机启动的命令

exit 0
```
现在在通过向 `/etc/rc.local` 添加开机自启任务的命令，就可以了。
