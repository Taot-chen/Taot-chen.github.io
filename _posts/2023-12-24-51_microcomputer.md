---
layout: post
title: 51_microcomputer
date: 2023-12-24
tags: [hardware]
author: taot
---

## 51 单片机

### 一、51 单片机 开发环境配置，vscode+SDCC

编辑器、编译器

最常用的集成开发环境 keil c51

#### 1、vscode + SDCC 开发环境搭建

vscode + 插件（ 或者 PlatformIO IDE）

* EIDE 的使用：详细自学

* PlatformIO IDE：详细自学

* vscode + SDCC 51 单片机开发环境搭建

    *   完全开源
    *   基于 vscode 写代码很舒服
    *   不支持在线调试
    *   更换或者添加源文件需要编写 Makefile 文件

    --> 环境搭建

    * vscode 安装完之后，添加 C/C++ 插件

    * 安装 SDCC

        * 安装的路径不能有空格

        * 安装完成之后，通过在 cmd 输入 `sdcc -v`，出现 sdcc 的版本信息，说明安装完成

            ![sdcc版本信息](https://image-hosting-taot.oss-cn-shanghai.aliyuncs.com/markdown-image202302110148257.png)

            

    * 安装 make，并加入环境变量

        ![make添加至环境变量](https://image-hosting-taot.oss-cn-shanghai.aliyuncs.com/markdown-image202302110153235.png)

        * 安装完成之后，通过在 cmd 输入 `make`，出现 make 的相关信息，说明安装完成

            ![make相关信息](https://image-hosting-taot.oss-cn-shanghai.aliyuncs.com/markdown-image202302110157273.png)

    #### 2、使用 SDCC 编译第一个 CH549 代码

    *   `CH549_sdcc.H` 复制到 `sdcc安装目录/include/mcs51`
    *   通过 code 打开 例程文件夹
    *   打开 `./usr/main.c` 文件
    *   `终端 --> 新终端`
    *   在终端中输入 make，即可编译程序；输入 make clean，即可删除刚才编译的程序

### 二、通用输入输出 GPIO

* GPIO：general purpose input or output，通用型输入或输出端口

    *   可以将单片机的每个引脚通过程序设定为通用输入或通用输出，可以实现与外部通信、控制外部硬件或者采集外部硬件数据的功能
    *   寄存器选择相应的功能，寄存器是 CPU 内部用来存放数据的一些小型存储区域，用来暂时存放参与运算的数据和运算结果
    *   寄存器的四种功能:
        *   接受二进制代码
        *   储存二进制代码
        *   输出二进制代码
        *   清除二进制代码
    *   具有移位功能的寄存器称为移位寄存器

* 代码含义：把地址为 0xA0  的寄存器的值配置为 2

    ![代码含义1](https://image-hosting-taot.oss-cn-shanghai.aliyuncs.com/markdown-image202302120114466.png)

* 代码编译与下载

    * 打开下载软件 --> 选择相应的芯片型号 --> 设置好下载配置 --> 连接单片机并按单片机上的 download 键 --> 选择刚才编译好的文件 --> 点击下载 --> 显示第一号设备下载完成，即可看到单片机上的 LED 等开始闪烁，若没有闪烁，可以按单片机上的 RST 键

        ![程序下载](https://image-hosting-taot.oss-cn-shanghai.aliyuncs.com/markdown-image202302120132418.png)

### 三、时钟、定时器中断、外部中断

* 时钟：时钟信号，固定频率的脉冲

    * CH549，运行一个指令（led = !led），就需要一个时钟周期

    * 分频器：将时钟信号转换为原来的时钟信号的频率的整数分之一（1/2， 1/3）

    * 倍频器：将时钟信号转换为原来的时钟信号的频率的整数倍

    * CH549  的时钟系统与结构图：

        ![CH549的时钟系统与结构图](https://image-hosting-taot.oss-cn-shanghai.aliyuncs.com/markdown-image202302120139701.png)

* 定时器中断

    *   定时器：16位寄存器，工作时，每接收到一个脉冲信号，它都会在原来的寄存器的保存的值上 +1 
        *   例如，main 函数中调用另一个函数的代码
    *   定时器中断：触发条件，当定时器溢出的时候，会出发中断的这样一个外设

* 外部中断：触发条件，某个端口满足设定的电平条件就能触发

* 其他中断：Uart 中断，ADC 中断等

* 中断优先级：高优先级的中断会打断低优先级的中断，在执行完高优先级的中断之后，再去执行低优先级的中断

    ![中断优先级](https://image-hosting-taot.oss-cn-shanghai.aliyuncs.com/markdown-image202302120149918.png)

* CH549 单片机的中断向量表

    ![CH549中断优先表](https://image-hosting-taot.oss-cn-shanghai.aliyuncs.com/markdown-image202302120151666.png)

* 外部中断和定时器中断例程代码

### 四、PWM

*   PWM：pulse width modulation，脉冲宽度调制
    *   通过控制高电平所占的时间，来达到控制功率的目的
    *   手机屏幕、电机控制
    *   PWM 属性：
        *   占空比：一个周期内高电平占整个周期的时间比例，反映了功率的控制情况，占空比越高功率越高
        *   频率：周期的倒数
*   Makefile 文件配置

### 五、通信：Uart (串口) ， i2c，spi

* 单片机常用通信协议：Uart， i2c，spi

* Uart（串口通信）：通用异步收发器，是一种串行、异步、全双工的通信协议

    *   串口具有通信输出线（TX）和通信接受线（RX）
    *   波特率：度量信息传输的速度，每秒传送的码元符号的个数，因此必须保证输出端和接收端的波特率是一致的

* 串口通信实验

    *   单片机的 TX 口连接串口转 USB 的 RX， 单片机的 RX 口连接串口转 USB 的 TX，这样可以实现数据回环的效果，即电脑发送给单片机的数据，单片机又会发送回给电脑

* SPI ：串行外设接口，一种高速的、全双工的、同步的串行通讯总线

    * 传输速率大于 Uart 和 i2c，同时可以进行发送和接收

    * spi 中有一根时钟线，无论输入还是输出，都需要通过时钟线

    * spi 通信至少需要四根线来实现：

        * MOSI：主设备数据输出，从设备数据输入

        * MISO：主设备数据输入，从设别数据输出

        * SCLK：时钟信号，由设备产生

        * CS：片选使能信号，由主设备产生

            ![spi原理](https://image-hosting-taot.oss-cn-shanghai.aliyuncs.com/markdown-image202302130154591.png)

    ### 六、ADDA（模拟信号与数字信号的互相转换）

    *   ADC：模数转换器，读取一个引脚模拟电压的大小
    *   DAC：数模转换器