---
layout: post
title: c_basic_9_unix_linux_introduction
date: 2023-12-27
---

## Unix/Linux操作系统

### 1、Unix/Linux操作系统介绍

#### 1.1、操作系统的作用

1）操作系统的目标

* 方便：使计算机系统易于使用
* 有效：以更有效的方式使用计算机系统资源
* 扩展：方便用户有效开发、测试、引进新功能

2）操作系统的地位

* 操作系统在计算机系统中有承上启下的地位，向下封装硬件，向上提供操作接口

#### 1.2、Unix/Linux操作系统介绍

1）Unix 家族

* 1965年，贝尔实验室，MULTICS 操作系统，失败
* 1969年，Unix 之父，Unics -> Unix，B语言和汇编语言
* 1971年，Unix 之父，C语言，C语言重写 Unix
* 1974年，Unix 流行开来
* 1980年，BSD Unix 和 AT&T 的 Unix
* 1982年，Unix System III，不再开原
* 1992-2001年，版权问题，两个 Unix 分枝逐渐衰败

2）Linux 家族

* Minix（mini-Unix），教学使用
* 1990年，Linus，1991年，Linux 内核正式发布
* Linux 系统的发展五个支柱：Unix 操作系统、minix 操作系统、GNU 计划、PUSIX 标准和互联网
* GNU 计划：GNU is Not Unix，包括：EMACS编辑系统、Bash shell程序、GCC、GDB等开发工具
* 1992年，Linux 和 GNU 软件结合 -> GNU/Linux（简称 Linux）
* POSIX 标准：操作系统应该为应用程序提供的接口标准，提高通用性和可移植性

3）Linux 的两类用户

* 知道自己在用 Linux：Linux 电脑系统
* 不知道自己在用 Linux：处理器、安卓内核等

4）Linux 的远亲

* macOS -> Darwin -> BSD -> unix

5）Linux 和 Unix 的联系

* Unix 是工作站上最常用的操作系统，多用户、多任务、实时，但是昂贵
* Linux，类 Unix，免费，Unix兼容的

6）Linux 内核及发行版本介绍

* 内核：运行程序和管理磁盘、打印机等一那件设备的核心程序，提供了一个在裸设备和应用程序之间的抽象层
* Linux 发行版：通常包含了桌面环境、办公套件、媒体播放器、数据库等应用软件

7）Unix/Linux 开发应用领域

* Unix/Linux 服务器
* 嵌入式 Linux 系统
* 桌面应用
* 电子政务

### 2、文件系统

#### 2.1、目录和路径

1）目录

* 目录是一组相关文件的集合
* 一个目录下面除了可以存放文件之外，还可以存放其他目录，即包含子目录
* 在确定文件、目录位置时，DOS 和 Linux/Unix 都采用 “路径名+文件名” 的方式。路径反映的是目录与目录之间的关系

2）路径

* Unix/Linux 路径由到达定位文件的目录组成。在Unix/Linux 中使用正斜杠 "/" 分割路径中的目录，在 DOS 中使用反斜杠 "\" 来分割。
* 相对路径和绝对路径：
    * 绝对路径：从根目录开始
    * 相对路径：目标目录相对于当前目录的位置。 "." 代表当前目录，".."代表当前目录的上级目录

#### 2.2、文件系统

1）Linux 和 Windows 文件系统的区别

* Windows 下目录都是起始于各个驱动器盘符（ A 盘和 B 盘是以前的软盘，现在不使用软盘，因此都从 C 盘开始）
* Linux 下没有驱动器盘符，只有目录，都是起始于相同的根目录 "/"

2）Linux 目录结构

* /：根目录
* /bin：/usr/bin：可执行二进制文件的目录，例如常用命令：ls、tar、mv、cat 等都在该目录下
* /boot：放置 Linux 启东市需要用到的一些文件，如内核文件、系统引导管理器等
* /dev：存放设备文件
* /etc：存放系统配置文件
* /home：系统默认的用户家目录
* /lib：/usr/lib：/usr/local/lib：系统使用的函数库目录
* /lost+fount：系统产生错误时，会讲一些遗失的片段放置在该目录下
* /mnt：/media：光盘的默认挂载点
* /opt：主机安装额外软件的安装目录
* /proc：存放系统核心数据、外部设备、网络状态等
* /root：系统管理员 root 的家目录
* /sbin：/use/sbin：/usr/local/sbin：放置系统管理员可以使用的可执行命令
* /tmp：一般用或者正在执行的程序临时存放文件的目录
* /srv：服务启动之后需要访问的数据目录
* /usr：应用程序的存放目录
* /var：放置系统执行过程中经常变化的文件

#### 2.3、一切皆文件

1）一切皆文件

* Unix/Linux 对数据文件、程序文件、设备文件、网络文件等的关林都抽象为文件，使用统一的方式进行管理
* 文件不通过后缀拓展名进行区分，可以没有拓展名，通过文件类型进行区分

2）文件分类

通常，Unix/Linux 中常用的文件类型有五种：普通文件、目录文件、设备文件、管道文件、链接文件

* 普通文件：存放数据、程序等信息的文件，一般包括文本文件、数据文件、可执行的二进制文件
* 目录文件：Unix/Linux 系统将目录看成是一种特殊的文件，利用它构成文件系统的树型结构
* 设备文件：Unix/Linux 系统把每一个设备都映射成一个设备文件，分为字符设备文件和块设备文件。字符设备的存取以字符为单位，块设备的存取以字符块为单位
* 管道文件：Unix/Linux 系统中用于进程间通信
* 链接文件：类似于 Windows 下的快捷方式，分为软链接和硬链接

#### 2.4、文件权限

1）访问用户

通过访问用户方式分为三种：

* 只允许用户自己访问（所有者）
* 允许一个预先设定的用户组中的用户访问（用户组）
* 允许系统中的任何用户访问（其他用户）

2）访问权限

* 读权限：r
* 写权限：w
* 可执行权限：x

3）说明

* 文件权限由 10 个字母表示
* 第一个字母时文件类型，d 表示目录文件，- 表示普通文件，c 表示硬件字符设备文件，b 表示硬件块设备文件，s 表示管道文件，l表示软链接文件
* 后面九个字母分成三组，每组三个。第一组是所有者权限，第二组是用户组权限，第三组是其他用户权限。权限的三个字母为;rwx，分别代表：读、写、执行。相应的字母表示该类用户具有相应的权限，- 表示该类用户不具有相应的权限

### 3、常用命令

#### 3.1、概述

常用的 Linux 命令大概200多个，需要记住的大概 20%

#### 3.2、命令使用方法

1）Linux 命令格式

**command [-options] [parameter1] [parameter2]...** 

* command：命令名
* [ioptions]：选项，用来对命令进行控制，可以省略
* parameter：传给命令的参数，可以使零个或者多个

2）--help 选项查看命令帮助文档

* --help 选项：一般是 Linux 命令自带的帮助信息，不是所有的命令都自带这个选项
* 例如，查看 ls 命令的帮助信息：ls --help

3）man 命令查看帮助手册

* man 是 Linux 提供的一个手册，包含了大部分的命令、函数的使用说明
* man 中包含多个章节，可以指定不同的章节查看
* man command，可以查看当前命令的说明

4）使用技巧

* 自动补全命令：敲出命令前几个字符之后，按 Tab 可以自动补全命令
* 历史命令：上下键，history 命令

#### 3.3、常用命令

1）文件管理

* 查看文件信息：ls，list 缩写，查看当前目录下的的文件列表

    * ls ./子目录名：查看当前目录子目录的内容
    * ls -a：显示当前目录下所有文件，包含隐藏文件
    * ls -l：以列表形式查看当前目录下所有文件，包括隐藏文件
    * ls -h：配合 -l 以人性化方式显示文件大小
    * 还允许使用通配符来同时引用多个文件名

    |  通配符   |                             含义                             |
    | :-------: | :----------------------------------------------------------: |
    |     *     |                     代表文件名中所有字符                     |
    |  ls te*   |              查看当前目录下所有以 te 开头的文件              |
    | le *html  |             查看当前目录下所有以 html 结尾的文件             |
    |    ？     |                     代表文件名中任意字符                     |
    |  ls ?.c   |           只找点前面一个字符，并且以 .c 结尾的文件           |
    |    []     | 使用 [] 把字符组括起来，表示可以匹配字符组中任意一个，- 表示字符范围 |
    |   [abc]   |                表示匹配 a、b、c 中的任意一个                 |
    |   [a-f]   |                 表示匹配 a 到 f 中的任意一个                 |
    | ls [a-f]* |           查找以 a 到 f 之间任意一个字符开头的文件           |

    **注：要使用通配符字符，需要使用 \ 进行转义；处于方括号内的通配字符不需要转义。例如：ls \*a 表示查看文件名为 *a 的文件** 

2）输出重定向命令

* `>`：可以将命令的输出结果重定向到一个文件，如：ls > test.txt，可以将命令的输出保存到 test.txt 中，如果文件不存在，会创建文件，如果文件存在，**会覆盖其内容**
* `>>`：可以将命令的输出结果重定向到一个文件，如：ls >> test.txt，可以将命令的输出保存到 test.txt 中，如果文件不存在，会创建文件，如果文件存在，**会在文件末尾追加内容**

3）分屏显示

* more：查看内容过长而无法在一屏内显示时，会出现快速滚屏，无法看清。使用 more 命令可以一次只显示一页，空格可以翻页，q 退出

4）管道

* |：一个命令的输出通过管道作为另一个命令的输入
    * ls -al | more：就可以分屏显示文件列表

5）清屏

* clear
* 快捷键：Ctrl+l

6）切换工作目录

* cd 目标目录路径：切换到目标目录
* cd ..：回到上级目录
* cd：切换到当前用户的主目录，cd ~ 也可以达到相同的效果
* cd .：切换到当前目录
* cd -：切换至上一个进入的目录

**可以使用绝对路径，也可以使用绝对路径** 

**Linux 的目录路径大小写敏感** 

7）显示当前路径

* pwd：显示当前所在路径

8）创建目录

* mkdir：mkdir 目录名，可以创建名为目录名的目录
* mkdir 目录名1 目录名2 ……，可以创建多个目录
* mkdir -p 目录名1/目录名2，可以递归创建目录

9）删除目录：

* mdir：mdir 目录名，可以删除名为目录名的目录，删除时必须离开目录，且目录为空，否则会出错
* mdir 目录名1 目录名2……，可以删除多个目录
* mdir -p 目录名1/目录名2，可以递归删除目录

10）删除文件

* rm：rm 文件名，删除文件，且不可恢复

* rm 常用参数

    | 参数 |                       含义                       |
    | :--: | :----------------------------------------------: |
    |  -i  |               以进行交互的方式执行               |
    |  -f  |       强制删除，忽略不存在的文件，无需提示       |
    |  -r  | 递归地删除目录下的内容，删除文件夹时必须加此参数 |

11）建立链接文件

* ln 源文件 链接文件，建立硬链接，两个文件占用相同大小的硬盘空间，即使删除了源文件，链接文件依然存在
* ln -s 源文件 链接文件，建立软链接
* 软链接：不占用磁盘空间，源文件删除，则软链接失效
* 硬链接：只能链接普通文件，不能链接目录

12）查看或者合并文件内容

* cat：cat 文件名：显示文件内容
* cat 文件名1 文件名2 >> 文件名1，将文件名1和文件名2的内容合并存放到文件名1中

13）拷贝文件

* cp：cp 源文件 目标文件，把源文件拷贝到目标文件中

* 常用选项

    | 选项 |                             含义                             |
    | :--: | :----------------------------------------------------------: |
    |  -a  | 用于拷贝目录时，拷贝时保留链接、文件属性，并递归复制目录，目标文件具有原文件的所有属性 |
    |  -f  |                覆盖已经存在的目标文件而不提示                |
    |  -i  |     交互式复制，在覆盖目标文件之前，给出提示要求用户确认     |
    |  -r  | 若给出的源文件是目录文件，则递归复制该目录下的所有文件和子目录，且目标文件也必须是一个目录文件 |
    |  -v  |                         显示拷贝进度                         |

14）移动文件

* mv：mv 源文件 目标文件，将源文件剪切为目标文件

* 常用选项：

    | 选项 |                      含义                      |
    | :--: | :--------------------------------------------: |
    |  -f  |      禁止交互式操作，如有覆盖也不给出提示      |
    |  -i  | 交互式操作，如果存在覆盖，给出提示要求用户确认 |
    |  -v  |                  显示移动进度                  |

15）获取文件类型

* file：file 文件名，显示文件类型信息

16）归档管理

* tar：tar [参数] 档案文件名 文件

* tar 命令比较特殊，其参数前面可以加 -，也可以不加

* 待归档文件可以有多个，依次列出即可

* 常用归档命令格式：tar -cvf 档案名.tar 文件

* 常用解档命令格式：tar -xvf 待解档文件

* 常用参数：

    | 参数 |                      含义                      |
    | :--: | :--------------------------------------------: |
    |  -c  |           生成归档文件，创建打包文件           |
    |  -v  |        列出归档解档的详细过程，显示进度        |
    |  -f  | 指定方案文件名称，这个参数必须放在所有参数最后 |
    |  -t  |              列出档案中包含的文件              |
    |  -x  |                  揭开档案文件                  |

    **注：只有参数 f 需要放在最后面，其他的参数没有顺序要求**

17）文件压缩解压

**gzip** 

* gzip：gzip [选项] 被压缩文件名

* tar 命令和 gzip 命令结合使用实现文件打包、压缩

* tar 只负责文件打包，但不压缩，用 gzip 压缩后的 tar 打包文件，其名称一般为：XXX.tar.gz

* 常用选项

    | 选项 |      含义      |
    | :--: | :------------: |
    |  -d  |      解压      |
    |  -r  | 压缩所有子目录 |

* 压缩已归档文件：gzip -r XXX.tar

    * 原本的 XXX.tar 文件就被 XXX.tar.gz 覆盖了

* 常用解压缩命令格式：gzip -d XXX.tar.gz

    * 原本的 XXX.tar.gz 文件就被 XXX.tar 覆盖了
    * 解档：tar -xvf XXX.tar

* 一步归档压缩：tar -zcvf xxx.tar.gz 1.c 2.c 3.c

    * 1.c 2.c 3.c 归档压缩为xxx.tar.gz

* 一步解压缩解档：tar -xzvf xxx.tar.gz

**bzip2** 

* bzip2：和 gzip 作用一样，文件名一般为：xxx.tar.bz2
    * 压缩：tar -cjvf xxx.tar.bz2 待压缩文件名们
    * 解压：tar -xjvf xxx.tar.bz2

**zip、unzip** 

* zip：zip 目标文件 文件名们
    * 通过 zip 压缩文件的目标文件不需要指定扩展名，默认为 zip
* 压缩：zip [-r] 目标文件 源文件们
    * 生成：目标文件.zip
* 解压：unzip -d 解压后目录文件 压缩文件
    * 解压到`解压后目录文件`文件夹内

18）查看命令位置

* which：which 命令名，查看命令的存放位置
    * which ls，查看 ls 命令的存放位置

#### 3.4、用户、权限管理命令

1）查看当前用户：whoami

* whoami，查看当前系统当前账号的用户名
* 可以通过 cat /etc/passwd 查看系统用户信息
* su：切换至 root 用户
* su 用户名：切换至`用户名` 用户

2）退出登录账户：exit

* 如果是图形界面，exit 会关闭终端
* 如果是 ssh 远程登录，会退出登录账户
* 如果是切换后的登录用户，exit 会返回上一个登录用户

3）切换用户：su

* su 用户名：命令切换用户
* su - 用户名：也可以切换用户，并且在切换到目标用户之后，会自动切换到当前用户的主目录
* su：切换至 root 用户
* 设置 root 用户密码：sudo passwd

4）添加、删除组账号：groupadd、groupdel

* groupadd 用户名：新建的组账号
* groupdel 用户名：删除组账号，组内没有成员才可以删除组
* cat /etc/group：查看用户组

5）修改用户所在组：usermod

* usermod -g 用户组名 用户名：把用户添加到用户组内

6）添加用户账号：useradd

* useradd [参数] 新建用户账号

* adduser [参数] 新建用户账号

* 参数说明

    | 参数 |                             含义                             |
    | :--: | :----------------------------------------------------------: |
    |  -d  | 指定用户登录系统时的主目录。如果不指定，系统自动在 /home 目录下建立与用户名同名的目录为主目录 |
    |  -a  |                         自动建立目录                         |
    |  -g  |                          指定组名称                          |

7）删除用户：userdel

* userdel 用户名：删除用户，但不会自动删除用户的主目录
* userdel -r 用户名：删除用户，同时删除用户的主目录

8）查询用户登录情况：last

* 管理员可以通过 last 命令查看每位用户的登录情况，登录地址

9）修改文件权限：chmod

* chmod 修改文件权限有两种格式：字母法和数字法

    * 字母法：chmod u/g/o/a +/-/= rwx 文件名

        | 符号 |         含义          |
        | :--: | :-------------------: |
        |  u   |   user，文件所有者    |
        |  g   | group，同一用户组用户 |
        |  o   |  other，其他以外的人  |
        |  a   |     all，三者皆是     |
        |  +   |       增加权限        |
        |  -   |       撤销权限        |
        |  =   |       设定权限        |

        * 例如：chmod go -r a.txt		撤销 g 和 o 的读权限
        * chmod a+x b.txt                    为 a 增加 x 权限
        * chmod u=rw abc.txt               设定 u 的权限为 rw

    * 数字法：rwx 这些权限可以使用数字来代替，rwx 组成三位二进制，组成的值就是对应的数字值

    | 权限 |     数字代号      |
    | :--: | :---------------: |
    |  r   |     读权限，4     |
    |  w   |         2         |
    |  x   |         1         |
    |  -   | 不具有任何权限，0 |

    * 例如：chmod u=7,g=5,o=4 filename	与	chmod u=rwx,g=rx,o=r filename	等价
    * chmod 751 file
        * 文件所有者：rwx
        * 同组用户：rx
        * 其他用户：x

10）修改文件所有者：chown

* chown 用户名 文件名：将文件的所有者修改为指定用户

11）修改文件所属组：chgrp

* chgrp 用户组名 文件名：将文件的所属组修改为指定用户组

#### 3.5、系统管理

1）查看进程信息：ps

* 进程是一个具有一定独立功能的程序，他是操作系统动态执行的基本单元

* ps 命令常用的选项：

    | 选项 |                   含义                   |
    | :--: | :--------------------------------------: |
    |  -a  | 显示终端上的所有进程，包括其他用户的进程 |
    |  -u  |            显示进程的详细状态            |
    |  -x  |          显示没有控制终端的进程          |
    |  -w  |       显示加宽，以便显示更多的信息       |
    |  -r  |           只显示正在运行的进程           |

* top 命令：动态显示进程，还可以进行条件检索

2）终止进程：kill

* kill [-signal] pid：终止指定 pid 的进程
* kill 命令指定进程的进程号，需要配合 ps 命令使用
* signal 的值从 0 到 15，其中 9 为绝对终止，可以处理一般信号无法终止的进程

3）后台程序：&、jobs、fg

* 用户可以后台执行程序：命令 &
* 如果程序已经在执行，ctrl+z 可以将程序调入后台
* jobs：查看后台执行的程序
* fg 编号：将后台程序调出到前台，编号是通过 jobs 查看的编号

4）关机、重启：reboot、shutdown、init

|       命令        |             含义             |
| :---------------: | :--------------------------: |
|      reboot       |         重启操作系统         |
|  shutdown -r now  | 重启操作系统，并提示其他用户 |
|  shutdown -h now  |           立刻关机           |
| shutdown -h 20:35 |    系统在今天 20:35 关机     |
|  shutdown -h +10  |       系统十分钟后关机       |
|      init 0       |             关机             |
|      init 6       |             重启             |

5）字符界面和图形界面切换

* Redhat 平台下：

    | init 3 | 切换到字符界面 |
    | :----: | :------------: |
    | init 5 | 切换到图形界面 |

* 其他大部分平台：

    | ctrl+alt+F2 | 切换到字符界面 |
    | :---------: | :------------: |
    | ctrl+alt+F7 | 切换到图形界面 |

6）查看或配置网卡信息：ifconfig

* ifconfig：显示所有网卡的信息

    |  显示字段   |                             说明                             |
    | :---------: | :----------------------------------------------------------: |
    |    eth0     |                         网络接口名称                         |
    | Link encap  |                         链路封装协议                         |
    |   Hwaddr    |                       网口的 MAC 地址                        |
    |  Inet addr  |                           IP 地址                            |
    |    Bcast    |                           广播地址                           |
    |    Mask     |                           子网掩码                           |
    |     UP      |       网口状态识别，UP表示已经启用，DOWN 表示已经停用        |
    |  BROADCAST  |                广播标识，标识广播是否支持广播                |
    |   RUNNING   |          传输标识，标识网口是否已经开始传输分组数据          |
    |  MULTICAST  |                多播标识，标识网口是否支持多播                |
    | MTU, Metric | MTU：最大传输单位，单位：字节；Metric：度量值，用户 RIP 建立网络路由用 |
    |  RX bytes   |                       接收数据字节统计                       |
    |  TX bytes   |                       发送数据字节统计                       |

* ifconfig 还可以配置网络参数;

    * ifconfig 网口名称 [地址协议参数] [address] [参数]
    * 地址协议参数：inet，表示 IPv4；inet6：表示 IPv6
    * 例如：ifconfig eth0 inet 192.168.10.254 netmask 255.255.255.0 up

* 常用参数

    |        参数        |       功能       |
    | :----------------: | :--------------: |
    |         -a         | 显示所有网口状态 |
    |   inet [IP 地址]   |   设置 IP 地址   |
    | netmask [子网掩码] |   设置子网掩码   |
    |         up         |     启用网口     |
    |        down        |     关闭网口     |

* ifconfig 配置的网络参数存放在内存中，关机重启就失效了。如果需要持久有效，需要修改网口配置文件：

    * redhat：/etc/sysconfig/network-scripts/ifcfg-eth0 文件：

        ```bash
        IPADDR=IP 地址
        GATEWAY=默认网关
        ```

    * Ubuntu：/etc/NetworkManager/system-connections/Wired connection1 文件：

        ```bash
        [ipv4]
        method=manual
        addresses1=IP 地址;24;默认网关;
        ```

7）测试远程主机连通性：ping

* ping [参数] 远程主机 IP 地址
* ping 通过 ICMP 协议向远程主机发送 ECHO_REQUEST 请求，期望主机回复 ECHO_REPLAY 消息
* 通过 ping 命令可以检查是否与远程主机建立了 TCP/IP 连接

### 4、编辑器

#### 4.1、gedit 编辑器

Linux 下的一个文本编辑器

#### 4.2、vi 编辑器

1）vi 介绍：编辑器之神

2）vi 基本操作

* vi filename：打开或者新建文件

2）vi 常用命令

**在命令模式下进入插入模式：**

* a：光标位置右边插入文字
* i：光标位置当前处插入文字
* o：光标位置下方开启新行
* O：光标位置上方开启新行
* I：光标所在行首插入文字
* A：光标所在行尾插入文字

**vi 的退出：**

* shitf+z+z：保存退出
* `:wq`：保存退出
* `:x`：保存退出
* `:w filename`：保存到指定文件
* `:q`：退出，如果文件修改但是没有保存，会提示无法退出
* `:q！`：退出，不保存
* `:！命令`：暂时离开 vi，执行命令

**vi 的删除和修改功能：**

* [n]x：删除光标后 n 个字符
* [n]X：删除光标前 n 个字符
* D：删除光标所在位置开始到此行行尾的字符
* [n]dd：删除从当前行开始的 n 行（实际上是剪切）
* [n]yy：复制从当前行开始的 n 行
* p：把粘贴板上的内容插入到当前行
* dG：删除光标所在行到文件尾的所有字符
* J：合并两行（将下一行和当前行合并，用空格连接）
* .：执行上一次操作
* u：撤销前一个命令

**vi 的定位和查找功能：**

* ctrl+f：向前滚动一个屏幕
* ctrl+b：向后滚动一个屏幕
* gg：到文件第一行行首
* G：到文件最后一行行首
* `:$`：到文件最后一行行首
* [n]G 或者[n]gg：到指定行，n 为目标行数
* /字符串：查找指定字符串
* n：寻找下一个
* N：回到前一个
* ？：寻找上一个
* /~字符串：寻找以字符串开始的行
* /字符串$：寻找以字符串结尾的行
* /a.b：查找字符串 a 任意字符 b

**vi 的替换和设置指令：**

* r：替换当前光标字符（很少用）
* `:r 文件名`:在光标当前位置下一行载入另一个文件
* `:s/p1/p2/g`：将当前行中所有的 p1 用 p2 替换
* `:g/p1/s/p2/g`：将当前文件中所有的 p1 用 p2 替换
* `:n1,n2s/p1/p2/g`：将 n1 到 n2 行中所有的 p1 用 p2 替换
* `:set ic`：搜寻时不区分大小写
* `:set noic`：搜寻时区分大小写
* `:set nu`：显示行号
* `:set nunu`：不显示行号

### 5、远程操作

#### 5.1、ssh介绍

* ssh：Source shell

#### 5.2、远程登录

1）Linux 平台相互远程

```bash
ssh -l username hostip
```

2）Windows 远程登录 Linux

Xmanager、Xshell、Xftp……

#### 5.3、远程文件传输

### 6、webserver 搭建
