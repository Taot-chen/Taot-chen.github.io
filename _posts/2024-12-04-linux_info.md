---
layout: post
title: linux_info
date: 2024-12-04
tags: [linux]
author: taot
---

## linux查看硬件信息

### 1 CPU 

```bash
# 查看 CPU 详细信息
# 总核数 = 物理CPU个数 X 每颗物理CPU的核数 
# 总逻辑CPU数 = 物理CPU个数 X 每颗物理CPU的核数 X 超线程数
cat /proc/cpuinfo


# 查看物理CPU个数
cat /proc/cpuinfo| grep "physical id"| sort| uniq| wc -l

# 查看每个物理CPU中core的个数(即核数)
cat /proc/cpuinfo| grep "cpu cores"| uniq

# 查看逻辑CPU的个数
cat /proc/cpuinfo| grep "processor"| wc -l

# 查看CPU信息（型号）
cat /proc/cpuinfo | grep name | cut -f2 -d: | uniq -c
```

### 2 内存

```bash
# 查看当前内存大小，已用空间等
sudo cat /proc/meminfo

# 查看内存型号、频率
sudo dmidecode -t memory
```
