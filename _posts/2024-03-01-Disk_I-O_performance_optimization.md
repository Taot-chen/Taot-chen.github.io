---
layout: post
title: Disk_I-O_performance_optimization
date: 2024-03-01
tags: [OS]
author: taot
---

## 磁盘IO性能优化


### 1 IO基准测试

优化之前，先确定IO性能优化的目标。换句话说，要先知道这些IO性能指标(比如IOPS、吞吐量、延迟等)，要达到多少才合适。IO性能指标是没有具体标准的，根据应用场景、使用的文件系统和物理磁盘等不同，这些性能指标和需求都会有差异。

为了更客观合理的评估优化效果，我们首先应该对磁盘和文件系统进行基准测试，得到文件系统或者磁盘IO的极限性能。

#### 1.1 fio(flexible IO Tester)

fio(flexible IO Tester)是最常用的文件系统和磁盘IO性能基准测试工具。它提供了大量的可定制化选项，可以用来测试，裸盘或者文件系统在各种场景下的IO性能，包括了不同块大小、不同IO引擎以及是否使用缓存等场景。

工具安装：
```bash
sudo apt-get install -y fio
```

安装完成后，就可以执⾏ man fio 查询它的使⽤⽅法。

比较常见的场景用法：
```bash
#随机读
fio -name=randread -direct=1 -iodepth=64 -rw=randread -ioengine=libaio -bs=4k -size=1G
#随机写
fio -name=randwrite -direct=1 -iodepth=64 -rw=randwrite -ioengine=libaio -bs=4k -size=1G
#顺序读
fio -name=read -direct=1 -iodepth=64 -rw=read -ioengine=libaio -bs=4k -size=1G -numjobs=
#顺序写
fio -name=write -direct=1 -iodepth=64 -rw=write -ioengine=libaio -bs=4k -size=1G -numjob

```

参数说明：
* direct，表示是否跳过系统缓存。上⾯示例中，设置的 1 ，就表示跳过系统缓存。
* iodepth，表示使⽤异步 I/O（asynchronous I/O，简称AIO）时，同时发出的 I/O 请求上限。在上⾯的示例中，设置的是64。
* rw，表示 I/O 模式。示例中， read/write 分别表示顺序读/写，⽽ randread/randwrite 则分别表示随机读/写。
* ioengine，表示 I/O 引擎，它⽀持同步（sync）、异步（libaio）、内存映射（mmap）、⽹络（net）等各种 I/O 引擎。上⾯示例中，设置的 libaio 表示使⽤异步 I/O。
* bs，表示 I/O 的⼤⼩。示例中，设置成了 4K（这也是默认值）。
* filename，表示⽂件路径，当然，它可以是磁盘路径（测试磁盘性能），也可以是⽂件路径（测试⽂件系统性能）。示例中，设置成了磁盘 /dev/sdb。不过注意，⽤磁盘路径测试写，会破坏这个磁盘中的⽂件系统，所以在使⽤前，⼀定要事先做好数据备份。


#### 1.2 blktrace+fio 的组合使⽤

通常情况下，应用程序的IO都是读写并行的，而且每次的IO大小也不一定相同。所以，前面的这几种场景，并不能精确模拟应用程序的IO模式。

fio支持IO的重放。借助blktrace，在配合上fio，就可以实现对应用程序IO模式的基准测试。可以先用blktrace，记录磁盘设备的IO访问情况；然后使用fio，重放blktrace的记录。
```bash
# 使⽤blktrace跟踪磁盘I/O，注意指定应⽤程序正在操作的磁盘
$ blktrace /dev/sdb
# 查看blktrace记录的结果
# ls
sdb.blktrace.0 sdb.blktrace.1
# 将结果转化为⼆进制⽂件
$ blkparse sdb -d sdb.bin
# 使⽤fio重放⽇志
$ fio --name=replay --filename=/dev/sdb --direct=1 --read_iolog=sdb.bin
```

这样，就通过 blktrace+fio 的组合使⽤，得到了应⽤程序 I/O 模式的基准测试报告。


### 2 IO 性能优化

得到 I/O 基准测试报告后，然后找出 I/O 的性能瓶颈并优化，就是⽔到渠成的事情了。可以从应用程序、文件系统以及磁盘角度来优化IO。

#### 2.1 应用程序优化

应用程序处于整个IO栈的最上端，它可以通过系统调用，来调整IO模式（顺序/随机、同步/异步），同时，它也是IO数据的最终来源。常见优化方法：

* 可以使用追加写代替随机写，减少寻址开销，加快IO写的速度
* 可以借助缓存IO，充分利用系统缓存，降低实际IO的次数
* 可以在应用程序内部构建自己的缓存，或者用redis这类外部缓存系统。
  * 一方面，能够在应用程序内部，控制缓存的数据和生命周期；
  * 另一方面，也能降低其他应用程序使用缓存对自身的影响。
  * C标准库提供的fopen、fread等函数，都会利用标准库的缓存，减少磁盘的操作。如果直接使用open、read等系统调用时，就只能利用操作系统提供的页缓存和缓冲区等，而没有库函数的缓存可用。
* 在需要频繁读写同一块磁盘空间时，可以用mmap代替read/write，减少内存的拷贝次数
* 在需要同步写的场景中，尽量将写请求合并，而不是让每个请求都同步写入磁盘，即可以利用fsync() 取代 O_SYNC。
* 在多个应用程序共享相同磁盘时，为了保证IO不被某个应用完全占用，推荐使用cgroups的IO子系统，来限制进程/进程组的IOPS以及吞吐量
* 在使⽤ CFQ 调度器时，可以⽤ ionice 来调整进程的 I/O 调度优先级，特别是提⾼核⼼应⽤的 I/O 优先级。ionice ⽀持三个优先级类：Idle、Best-effort 和 Realtime。其中， Best-effort 和 Realtime 还分别⽀持 0-7 的级别，数值越⼩，则表示优先级别越⾼。

#### 2.2 文件系统优化

应用程序访问普通文件时，实际是由文件系统间接负责文件在磁盘中的读写。所以，跟文件系统中相关的也有很多优化IO性能的方式：

* 可以根据实际负载场景的不同，选择最合适的文件系统
  * Ubuntu 默认使⽤ ext4 ⽂件系统，⽽ CentOS 7 默认使⽤ xfs ⽂件系统。相⽐于 ext4 ，xfs ⽀持更⼤的磁盘分区和更⼤的⽂件数量，如 xfs ⽀持⼤于16TB 的磁盘。但是 xfs ⽂件系统的缺点在于⽆法收缩，⽽ ext4 则可以。
* 在选好⽂件系统后，还可以进⼀步优化⽂件系统的配置选项，包括⽂件系统的特性（如ext_attr、dir_index）、⽇志模式（如journal、ordered、writeback）、挂载选项（如noatime）等等。
  * 使⽤tune2fs 这个⼯具，可以调整⽂件系统的特性（tune2fs 也常⽤来查看⽂件系统超级块的内容）。
  * 通过 /etc/fstab，或者 mount 命令⾏参数，我们可以调整⽂件系统的⽇志模式和挂载选项等。
* 可以优化⽂件系统的缓存。
  * 可以优化 pdflush 脏⻚的刷新频率（⽐如设置 dirty_expire_centisecs 和dirty_writeback_centisecs）以及脏⻚的限额（⽐如调整 dirty_background_ratio 和 dirty_ratio等）。
  * 可以优化内核回收⽬录项缓存和索引节点缓存的倾向，即调整vfs_cache_pressure（/proc/sys/vm/vfs_cache_pressure，默认值100），数值越⼤，就表示越容易回收。
* 在不需要持久化时，还可以利用内存文件系统fs，以获得更改的IO性能。
  * tmpfs把数据直接保存在内存中，而不是在磁盘中。/dev/shm/ ，就是⼤多数 Linux 默认配置的⼀个内存⽂件系统，它的⼤⼩默认为总内存的⼀半。

#### 2.3 磁盘优化

数据的持久化存储，最终还是要落在具体的物理磁盘中，同时，磁盘也是整个IO栈的最底层。从磁盘角度出发，也有很多有效的优化方法：

* 最简单有效的优化方法，就是换用更好的磁盘，比如用SSD替代HDD
* 可以使用RAID，把多个磁盘组成一个逻辑磁盘，构成冗余独立磁盘阵列。这样做既可以提高数据的可靠性，又可以提高数据的访问性能
* 针对磁盘和应用程序IO模式的特性，我们可以选择最合适的IO调度算法。比如，SSD和虚拟机中的磁盘，通常用noop调度算法；而数据库应用建议使用deadline算法
* 可以对应用程序的数据，进行磁盘级别的隔离。比如，我们可以为日志、数据库等IO压力比较重的应用，配置单独的磁盘
* 在顺序读比较多的场景中，我们可以增大磁盘的预读数据，⽐如，你可以通过下⾯两种⽅法，调整 /dev/sdb 的预读⼤⼩。
  * 调整内核选项 /sys/block/sdb/queue/read_ahead_kb，默认⼤⼩是 128 KB，单位为KB。
  * 使⽤ blockdev ⼯具设置，⽐如 blockdev --setra 8192 /dev/sdb，注意这⾥的单位是 512B（0.5KB），所以它的数值总是read_ahead_kb 的两倍。
* 可以优化内核块设备IO的选项。
  * 可以调整磁盘队列的⻓度 /sys/block/sdb/queue/nr_requests，适当增⼤队列⻓度，可以提升磁盘的吞吐量（当然也会导致 I/O 延迟增⼤）。
* 磁盘本身出现硬件错误，也会导致IO性能急剧下降，所以发现磁盘性能急剧下降时，先确定，磁盘本身是不是出现了硬件错误。
  * 可以查看 dmesg 中是否有硬件 I/O 故障的⽇志。
  * 可以使⽤ badblocks、smartctl等⼯具，检测磁盘的硬件问题，或⽤ e2fsck 等来检测⽂件系统的错误。如果发现问题，可以使⽤ fsck 等⼯具来修复。

*磁盘和文件系统的IO，通常是整个系统中最慢的一个模块。所以，在优化IO问题时，除了可以优化IO的执行流程，还可以接入更快的内存、网络、CPU等，减少IO调用*

