---
layout: post
title: raspi5_kernel_source_code
date: 2024-11-05
tags: [raspi]
author: taot
---

## 树莓派linux内核源码编译

Raspberry Pi 内核 托管在 [GitHub](https://github.com/raspberrypi/linux) 上；更新滞后于上游 [Linux内核](https://github.com/torvalds/linux)，Raspberry Pi 会将 Linux 内核的长期版本整合到 Raspberry Pi 内核中。


### 1 构建内核

操作系统随附的默认编译器和链接器被配置为构建在该操作系统上运行的可执行文件。**原生编译**使用这些默认编译器和链接器。**交叉编译**是为运行编译过程的目标之外的目标编译代码的过程。Raspberry Pi 内核的交叉编译允许从 32 位操作系统构建 64 位内核，反之亦然。也可以从 Raspberry Pi 以外的设备交叉编译 32 位或 64 位 Raspberry Pi 内核。

#### 1.1 下载内核源代码

下载最新 Raspberry Pi 内核的源代码：
```bash
git clone --depth=1 https://github.com/raspberrypi/linux
```

> 上面的 git clone 命令下载的是当前活动分支，从这个分支构建 Raspberry Pi 操作系统镜像，没有任何历史记录。省略 --depth=1可以下载整个版本库，包括所有分支的完整历史。
> 要下载不带历史记录的不同分支，在上述命令中添加 --branch 选项，并将 <branch> 替换为要下载的分支名称
> ```bashgit clone --depth=1 --branch <branch> https://github.com/raspberrypi/linux```


#### 1.2 原生构建内核

##### 1.2.1 安装编译依赖项

```bash
sudo apt install bc bison flex libssl-dev make
```

##### 1.2.2 构建配置

在构建内核时应用默认配置。准备默认配置，需要根据 Raspberry Pi 型号运行相应命令，这里使用的是树莓派5，命令为：
```bash
cd linux
KERNEL=kernel_2712
make bcm2712_defconfig
```

##### 1.2.3 使用 LOCALVERSION 自定义内核版本

为防止内核覆盖 `/lib/modules` 中的现有模块，并在 `uname` 输出中说明运行的是自己的内核，调整 `LOCALVERSION`。

修改 `.config` 中的以下一行:
```bash
CONFIG_LOCALVERSION="-v7l-MY_CUSTOM_KERNEL"
```

##### 1.2.4 构建

构建内核。这一步可能需要很长时间, 取决于 Raspberry Pi 型号:
```bash
# 构建 64 位内核
make -j6 Image.gz modules dtbs

# 构建 32 位内核
make -j6 zImage modules dtbs
```

##### 1.2.5 安装内核

将内核模块安装到启动媒体上：
```bash
sudo make -j6 modules_install
```

然后，将内核和设备树 blobs 安装到启动分区中，并备份原始内核。
> 如果不想在运行此命令的 Raspberry Pi 上安装刚编译好的内核，可将编译好的内核复制到另一个启动介质的启动分区，而不是 /boot/firmware/。

* 安装 64 位内核：
```bash
# 运行以下命令创建当前内核的备份镜像，安装新的内核镜像、覆盖层、README，并卸载分区
sudo cp /boot/firmware/$KERNEL.img /boot/firmware/$KERNEL-backup.img
sudo cp arch/arm64/boot/Image.gz /boot/firmware/$KERNEL.img
sudo cp arch/arm64/boot/dts/broadcom/*.dtb /boot/firmware/$KERNEL-backup.img
sudo cp arch/arm64/boot/dts/overlays/*.dtb* /boot/firmware/overlays/
sudo cp arch/arm64/boot/dts/overlays/README /boot/firmware/overlays/
```

* 安装 32 位内核：
```bash
# 创建当前内核的备份，然后安装新的内核映像
sudo cp /boot/firmware/$KERNEL.img /boot/firmware/$KERNEL-backup.img
sudo cp arch/arm/boot/zImage /boot/firmware/$KERNEL.img

# 根据内核版本，运行以下命令
# 对于版本 6.4 以下的内核：
sudo cp arch/arm/boot/dts/*.dtb /boot/firmware/
# 适用于 6.5 及以上版本的内核：
sudo cp arch/arm/boot/dts/broadcom/*.dtb /boot/firmware/

# 复制覆盖层和 README
sudo cp arch/arm/boot/dts/overlays/*.dtb* /boot/firmware/overlays/
sudo cp arch/arm/boot/dts/overlays/README /boot/firmware/overlays/
```

* 运行以下命令重启 Raspberry Pi 并运行新编译的内核
```bash
sudo reboot
```

#### 1.3 交叉编译内核

首先需要准备一台合适的 Linux 交叉编译主机，这里使用的是 ubuntu。

##### 1.3.1 安装所需的依赖项和工具链
```bash
sudo apt install bc bison flex libssl-dev make libc6-dev libncurses5-dev

# 安装 64 位工具链以构建 64 位内核，请运行以下命令
sudo apt install crossbuild-essential-arm64

# 安装 32 位工具链以构建 32 位内核，请运行以下命令
sudo apt install crossbuild-essential-armhf
```

##### 1.3.2 编译配置

在构建内核时应用默认配置。准备默认配置，需要根据 Raspberry Pi 型号运行相应命令，这里使用的是树莓派5，命令为：
```bash
cd linux
KERNEL=kernel_2712
make ARCH=arm64 CROSS_COMPILE=aarch64-linux-gnu- bcm2712_defconfig
```

##### 1.3.3 使用 LOCALVERSION 自定义内核版本

为防止内核覆盖 `/lib/modules` 中的现有模块，并在 `uname` 输出中说明运行的是自己的内核，调整 `LOCALVERSION`。

修改 `.config` 中的以下一行:
```bash
CONFIG_LOCALVERSION="-v7l-MY_CUSTOM_KERNEL"
```

##### 1.3.4 构建

* 构建 64 位内核
```bash
make ARCH=arm64 CROSS_COMPILE=aarch64-linux-gnu- 映像模块 dtbs
```

* 构建 32 位内核
```bash
make ARCH=arm CROSS_COMPILE=arm-linux-gnueabihf- zImage 模块 dtbs
```

##### 1.3.5 安装内核

构建内核后，需要将其复制到 Raspberry Pi 的启动媒体（可能是 SD 卡或 SSD）上并安装模块。

* 查找启动媒体
首先，运行 lsblk。然后，连接启动媒体。再次运行 lsblk；新设备代表启动媒体
例如我这里的输出为：
```bash
sdb
   sdb1
   sdb2
```
将这些分区挂载为 mnt/boot 和 mnt/root，调整分区代号以匹配启动媒体的位置：
```bash
mkdir mnt
mkdir mnt/boot
mkdir mnt/root
sudo mount /dev/sdb1 mnt/boot
sudo mount /dev/sdb2 mnt/root
```

* 安装
将内核模块安装到启动媒体上：
```bash
# 对于 64 位内核
sudo env PATH=$PATH make -j12 ARCH=arm64 CROSS_COMPILE=aarch64-linux-gnu- INSTALL_MOD_PATH=mnt/root modules_install

# 对于 32 位内核
sudo env PATH=$PATH make -j12 ARCH=arm CROSS_COMPILE=arm-linux-gnueabihf- INSTALL_MOD_PATH=mnt/root modules_install
```

* 将内核和设备树 blob 安装到启动分区，并备份原始内核

```bash
# 安装 64 位内核
sudo cp mnt/boot/$KERNEL.img mnt/boot/$KERNEL-backup.img
sudo cp arch/arm64/boot/Image mnt/boot/$KERNEL.img
sudo cp arch/arm64/boot/dts/broadcom/*.dtb mnt/boot/
sudo cp arch/arm64/boot/dts/overlays/*.dtb* mnt/boot/overlays/
sudo cp arch/arm64/boot/dts/overlays/README mnt/boot/overlays/
sudo umount mnt/boot
sudo umount mnt/root



# 安装 32 位内核
sudo cp mnt/boot/$KERNEL.img mnt/boot/$KERNEL-backup.img
sudo cp arch/arm/boot/zImage mnt/boot/$KERNEL.img
# 对于版本 6.4 以下的内核
sudo cp arch/arm/boot/dts/*.dtb mnt/boot/
#  6.5 及以上版本的内核
sudo cp arch/arm/boot/dts/broadcom/*.dtb mnt/boot/

sudo cp arch/arm/boot/dts/overlays/*.dtb* mnt/boot/overlays/
sudo cp arch/arm/boot/dts/overlays/README mnt/boot/overlays/
sudo umount mnt/boot
sudo umount mnt/root
```

最后，将启动媒体连接到 Raspberry Pi 并接通电源，运行新编译的内核。

### 2 配置内核

Linux 内核是高度可配置的, 以便根据自己的需要进行定制，例如启用新的或试验性的网络协议，或启用对新硬件的支持。最常见的配置方法是通过 make menuconfig 界面完成, 也可以手动修改你的 `.config` 文件:

#### 2.1 准备配置

menuconfig 工具需要 ncurses 开发头文件才能正常编译:
```bash
sudo apt install libncurses5-dev
```

接下来，下载内核源代码。

#### 2.2 menuconfig

编译并运行 menuconfig 工具:
```bash
make menuconfig

# 交叉编译 64 位内核：
make ARCH=arm64 CROSS_COMPILE=aarch64-linux-gnu- menuconfig
# 交叉编译 32 位内核：
make ARCH=arm CROSS_COMPILE=arm-linux-gnueabihf- menuconfig
```

简短编译后，menuconfig 将显示包含所有可配置选项的子菜单列表。第一次尝试时，不要轻易启用或禁用很多选项；这样很容易破坏配置。

### 3 内核补丁

在构建自定义内核时，可能希望在 Linux 内核中应用补丁或补丁集（patchsets）。

#### 3.1 确定内核版本

查看设备上当前运行的内核版本：
```bash
uname -r
# 6.6.51+rpt-rpi-2712
```

在打补丁之前，务必检查内核版本。在内核源代码目录下，运行以下命令查看内核版本:
```bash
head Makefile -n n

# 类似下面的输出：
# SPDX-License-Identifier: GPL-2.0
VERSION = 6
PATCHLEVEL = 1
SUBLEVEL = 38
EXTRAVERSION =
NAME = Pinguïn Aangedreven
```
这里可以看到版本为 6.1.38。

#### 3.2 应用补丁

补丁的应用取决于补丁的发布格式。开发者会以单个文件的形式发布大多数补丁。使用 patch 工具来打补丁。

以下命令将下载、解压缩并使用实时内核补丁：
```bash
wget https://www.kernel.org/pub/linux/kernel/projects/rt/6.1/patch-6.1.38-rt13-rc1.patch.gz
gunzip patch-6.1.38-rt13-rc1.patch.gz
cat patch-6.1.38-rt13-rc1.patch | patch -p1
```

有些开发者会以 邮箱格式 发布补丁，即一个包含多个补丁文件的文件夹。使用 Git 可以打上这些补丁:
```bash
git am -3 /path/to/patches/*
```
