---
layout: post
title: expand_swap_partition
date: 2024-01-28
tags: [tools]
author: taot
---

## ubuntu 增加 swap 空间大小

之前装系统的时候，使用了默认的分区方案，结果只有 2GB 的 swap 分区，机器只有这么点内存，平时使用经常出现内存不够用，又没有分配足够的交换空间，市场感觉到拮据，有必要增加一些 swap 空间大小。

### 1 查看系统内 swap 分区大小
```bash
~$ free -h
              total        used        free      shared  buff/cache   available
Mem:           125G        730M        124G        4.1M        690M        123G
Swap:          2.0G          0B        2.0G

```

可以看到 Swap 只有 2GB，下面扩大到 256GB。

查看交换分区路径：

```bash
sudo swapon --show
NAME      TYPE SIZE USED PRIO
/swapfile file   2G   0B   -2
```

### 2 创建新的 swap 文件

#### 2.1 找一个空间足够的目录用来存放swap文件

```bash
mkdir /swap
cd /swap\
sudo dd if=/dev/zero of=swapfile bs=1024 count=268435456
```

#### 2.2 把生成的文件转换成 Swap 文件

```bash
sudo fallocate -l 1G /swapfile
```

执行以下命令为 swapfile 文件设置正确的权限：

```bash
sudo chmod 600 /swapfile

sudo mkswap -f /swapfile
```

### 3 激活 swap 文件

```bash
sudo swapon /swapfile
```

再次查看 free -h 的结果:

```bash
~$ free -h
              total        used        free      shared  buff/cache   available
Mem:           125G        730M        124G        4.1M        690M        123G
Swap:          256G          0B        256G
```


### 4 调整 Swappiness 值

Swappiness 是一个 Linux 内核属性，用于定义 Linux 系统使用 SWAP 空间的频率。Swappiness 值可以从 0 至 100，较低的值会让内核尽可能少的使用 SWAP 空间，而较高的值将让 Linux Kernel 能够更加积极地使用 SWAP 分区。

Ubuntu 18.04 默认的 Swappiness 值为 60，可以使用如下命令来查看：

```bash
cat /proc/sys/vm/swappiness
```

值为 60 对于 Ubuntu 18.04 桌面还算行，但对于 Ubuntu Server 来说，SWAP 的使用频率就比较高了，所以可能需要设置较低的值。
将 swappiness 值设置为 40:

```bash
sudo sysctl vm.swappiness=40
```

如果要让设置在系统重启后依然有效，则必要在 /etc/sysctl.conf 文件中添加以下内容:

```bash
vm.swappiness=40
```

最佳 swappiness 值取决于系统的工作负载以及内存的使用方式。在调整的时候应该以小增量的方式来调整此参数，以查到最佳值。


### 5 移除SWAP分区

Ubuntu 18.04 要停用并删除 SWAP 文件，可按照下列步骤操作：

#### 5.1 停用 SWAP 空间

```bash
sudo swapoff -v /swapfile
```

#### 5.2 在 /etc/fstab 文件中删除有效 swap 的行

#### 5.3 删除 swapfile 文件

```bash
sudo rm /swapfile
```

扩大原有swap交换分区，需要先移除，然后重新创建添加。如果添加第二个swap分区，系统反而会变慢。
