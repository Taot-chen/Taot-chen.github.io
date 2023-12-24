---
layout: post
title: brushless_motor
date: 2023-12-24
---

## 无刷电机控制原理

### 1、无刷电机结构

![无刷电机结构](https://image-hosting-taot.oss-cn-shanghai.aliyuncs.com/markdown-image202304210041238.png)

直流无刷电机驱动：依靠改变电机定子线圈的电流交变频率和波形，在定子周围形成磁场，驱动转子永磁体转动，进而带动电机转起来

研究如何产生和改变输入定子三相的电流交变频率和波形：

*   硬件电路部分
*   软件控制部分：FOC

硬件电路部分：无法使用类似有刷电机的电刷这样的机械机构来实现电流换向，而是使用例如 MOS 这样的器件来实现

![无刷电机硬件控制原理](https://image-hosting-taot.oss-cn-shanghai.aliyuncs.com/markdown-image202304210051742.png)

![无刷电机硬件控制原理_2](https://image-hosting-taot.oss-cn-shanghai.aliyuncs.com/markdown-image202304210056114.png)

--> 对无刷电机的控制，实际上就是对这些 MOS  管的开关的控制，FOC 算法就是程序来控制这些 MOS 的开关，来实现对电机转速、扭矩等参数的控制

