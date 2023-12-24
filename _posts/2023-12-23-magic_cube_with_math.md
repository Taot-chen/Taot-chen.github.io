---
layout: post
title: magic_cube_with_math
date: 2023-12-23
---


## 一、魔方和数学建模（魔方状态和操作的数学表示）

### 1、笛卡尔坐标系安置

* 原点在魔方体中心
* 右手系，front --> x 轴，left --> y轴，up --> z轴

### 2、魔方和晶体学

* 短群运算

* 矢量标注、运算

* 魔方大小表示：

    * （1,0，-1）：三个数排列，27种排列，魔方 算上坐标原点的  中心块（000），一一对应

    ![image-20221104012646978](https://image-hosting-taot.oss-cn-shanghai.aliyuncs.com/markdown-image202211152357689.png)

### 3、魔方和矩阵

* 行列式，表示一个数；矩阵，表示一个操作，也表示一个状态

* 矩阵的乘法：行乘列得到一个元素

* 矩阵和矢量的运算：矩阵乘矢量，表示矩阵对矢量的一个操作

* 魔方有六个轴，六个可转动的面

* 转动矩阵三要素（魔方转动三要素）：转轴、转向、转角

* 转轴和转向的关系满足右手系

* 魔方转动矩阵：根据转动前后矢量坐标来计算出转动矩阵

    * 绕 z 轴逆时针转动 90 度：

        ![image-20221104014034246](https://image-hosting-taot.oss-cn-shanghai.aliyuncs.com/markdown-image202211152357711.png)

    * 绕 y 轴逆时针转动 90 度：

        ![image-20221104014123962](https://image-hosting-taot.oss-cn-shanghai.aliyuncs.com/markdown-image202211240020550.png)

    * 绕 yx轴逆时针转动 90 度：

        ![image-20221104014227114](https://image-hosting-taot.oss-cn-shanghai.aliyuncs.com/markdown-image202211240020375.png)

* **群的定义**：操作一个集合

    * 满足以下四个条件就构成一个群：
        * 任意两个操作的积还是集合内的一个操作
        * 集合内有一个恒等操作 I
        * 每一个操作 T 都有一个逆操作 $T^{-1}$ ，使得 $TT^{-1}=I$
        * 操作的乘法满足结合律
    * 魔法矩阵在做乘法的时候不满足交换律

* 魔方方程：

    * ![image-20221104015057983](https://image-hosting-taot.oss-cn-shanghai.aliyuncs.com/markdown-image202211240020635.png)

* 方位坐标系和左右手分工

    * ![image-20221104015142045](https://image-hosting-taot.oss-cn-shanghai.aliyuncs.com/markdown-image202211240020596.png)

* 魔方转动矩阵

    * X方向：
        * ![image-20221104015453327](https://image-hosting-taot.oss-cn-shanghai.aliyuncs.com/markdown-image202211152357222.png)
    * Y 方向：
        * ![image-20221104015521734](https://image-hosting-taot.oss-cn-shanghai.aliyuncs.com/markdown-image202211152357258.png)
    * Z方向：
        * ![image-20221104015550679](https://image-hosting-taot.oss-cn-shanghai.aliyuncs.com/markdown-image202211152357306.png)

* 8 个角块可以互相交换，12 个棱块可以互相交换，六个中心块，不可交换

* 魔方符号定义：色位

    * 前（F）、后（B）、左（L）、右（R）、上（U）、下（D）

* 转动的角块变化

* 转动的色块变化

    * 镜像处理：转动前存在 负号，需要进行镜像处理，再操作，操作完再镜像回来
    * 右手化处理：转动之后，必须是 ijk 的顺序，那么表示颜色的负号也需要进行相应的操作变换

    ![image-20221104022123245](https://image-hosting-taot.oss-cn-shanghai.aliyuncs.com/markdown-image202211152357346.png)

### 4、数学模型和程序设计

* N 阶魔方：
    * 小块数量：$N^3-(N-2)^3$
    * 奇数阶魔方：$h_{max}=(N-1)/2$     $h_{min}=0$
    * 偶数阶魔方：$h_{max}=N/2$     $h_{min}=1$
    * 方向指数和转层的关系
* 程序设计：上面的操作转换成程序语言
* 转层和转动矩阵：对于某一个特定的层，转动矩阵和 3 阶的一致，只是作用在不同的层上
    * ![image-20221104023353382](https://image-hosting-taot.oss-cn-shanghai.aliyuncs.com/markdown-image202211152357412.png)
* 计算机程序框架：
    * ![image-20221104023528977](https://image-hosting-taot.oss-cn-shanghai.aliyuncs.com/markdown-image202211152357484.png)

## 二、魔方与群论

* 魔方中的块分成三类：角块，有三种颜色；棱块，有两种颜色；面块，有一种颜色。

* 魔方中的旋转是 **非阿贝尔** 的，例如：$FR \ne RF$，即魔方群是非阿贝尔群。

* 操作的阶：对于一个还原的魔方，做任意一个固定的操作组合，重复有限次之后，魔方必定会被还原。这个有限次的重复次数，就是这个操作的阶。

* 群：元素和操作的整体，即为群。魔方群：魔方状态数和魔方操作的结合

* 魔方群是置换群的子群

* 置换群：置换，就是将一系列有顺序的元素重新排列。

* 一个可复原的三阶魔方的总变化数：
    $$
    N = \frac{8! \times 3^8 \times 12! \times 2^{12}}{2 \times 2 \times 3} \rightarrow 4.33 \times {10}^{19}
    $$

    * 八个角块，可以互换位置（8!），可以改变朝向（$3^8$），但是无法单独改变一个角块的朝向（$\frac{1}{3}$）

    * 十二个棱块，可以互换位置（12!），可以改变朝向（$2^{12}$），但是无法单独改变一个棱块的朝向（$\frac 1 2 $），也无法单独交换两个棱块（$\frac 1 2$）

    * 六个面块，无法改变状态

    * 二阶魔方：$N = 3.67 \times {10}^{6}$

    * 四阶魔方：$N = 7.40 \times {10}^{45}$

    * 五阶魔方：$N = 2.83 \times {10}^{74}$

    * 六阶魔方：$N = 1.57 \times {10}^{116}$

    * 七阶魔方：$N = 1.95 \times {10}^{160}$



## 三、魔方解法

### 1、降群法

降群法就是改变魔方的生成元，从而减少魔方可能的状态数。区别于常规的魔方解法从局部到整体的思路，降群法是直接从整体入手，减少魔方的状态数。

1）降群法思路

* 魔方有六个生成元：$G_0=<U,D,F,B,R,L>$ ，此时的群元素有 $4.33 \times 10^{19}$ 

* 通过对转动操作做出一定的限制，改变魔方的生成元：

    * 限制左右两个面的转动只能是 $180^o$（此时如果需要转动 $90^o$，必须在不影响棱块定向的基础上做逆操作），此时魔方的生成元为：$G_1=<U,D,F,B,R^2,L^2>$ ，此时的群元素有 $\frac{4.33 \times 10^{19}}{2^{11}} $ ，$\rightarrow$ **对好所有棱块方向**
    * 再限制魔方的前后面只能转动 $180^o$，此时魔方的生成元为：$G_2=<U,D,F^2,B^2,R^2,L^2>$ ，此时的群元素有 $\frac{4.33 \times 10^{19}}{2^{11} \times \frac{12!}{8! \times 4!} \times 3^7}$ $\rightarrow$ **将中间棱块放到中间层，并调整好角块朝向**
    * 再进一步 限制魔方的上下面只能转动 $180^o$，此时魔方的生成元为：$G_3=<U^2,D^2,F^2,B^2,R^2,L^2>$ ，此时的群元素有$\frac{4.33 \times 10^{19}}{2^{11} \times \frac{12!}{8! \times 4!} \times 3^7 \times {\left(\frac{8!}{4! \times 4!}\right)}^{2} \times 3 \times 2} = 663552$ $\rightarrow$ **调整F/B面为RO色，L/R面为BG色，并调整角块相对位置**
    * 剩下的魔方状态都是非常直观的，几乎可以通过简单的观察后凭借直觉复原，从而将魔方群降为单位元生成的群$G=<I>$  $\rightarrow$ **还原所有角块棱块**

    

### 2、二阶段算法（降群法的一个实现方案）

利用计算机使用常规的方法解魔方是非常容易的，甚至不需要借助基础的群论知识进行数学建模即可实现，但是高阶/高维魔方，这样的程序编写就变得困难了。另外，对于简单的三维三阶的魔方，当我们希望寻找一个尽可能短的解法的时候，这样的方案显得力不从心了，这是因为对于简单的三维三阶的魔方，就有 $4.33 \times 10^{19}$ 种状态数，比计算机的主频还高了快十个数量级。

为了解决状态数太多算不过来的问题，不难想到把还原的过程分成多个阶段依次处理。例如对于人类算法 CFOP，我们会把还原过程分成四个阶段。**二阶段算法顾名思义，就是把还原的过程分成了两个阶段。**

对于普通的三维三阶魔方，面块是不会改变位置的，只有角块和棱块需要还原。那么假设可以将三维三阶魔方的还原分为 还原角块和还原棱块两个阶段。那么，求解三维三阶魔方的问题就变成了两个相对简单的子问题，从而可以利用计算机快速求解。但是，这样会带来另一个问题，我们可以让两个子问题的求解都是最优的，但是合并子问题之后，解法就不是最优的了。这是显然的，因为对于第二阶段，需要在保证已经被还原的角块不被破坏的条件下还原棱块，这一阶段的解法对于整体的解法来说必然不是最优的。

因此，在第一阶段，不局限于局部的最优解法，而是根据解法长度从小到大依次找到更多的解法；依次尝试第一阶段解法，处理第二阶段的问题，找到整体最优解法。

$\rightarrow$ 二阶段解法的特点：

* 一定有解，且计算量可控，首个解法的计算量不会超过第一阶段和第二阶段各执行一次的最大计算量
* 计算时间越长，找到的解法的期望长度就越短，可以在计算时间和解的长度折中取舍
* 只要计算时间充分长，一定能够找到最优解

其实上面的两步是极不合理的：

* 角块的状态空间（$10^7$）远小于棱块的状态空间（$10^{11}$）
* 阶段二的搜索受限于阶段一的限制，阶段二很容易出现较长的解法

**真实的二阶段算法**

第一阶段：还原至 $<U,R^2,F^2,D,L^2,B^2>$ 的生成群；等价于调整魔方的所有块的色向正确，中间层的四个棱块槽位正确。第一阶段状态空间$2.2 \times 10 ^9$。此时的魔方群为 $<U,R^2,F^2,D,L^2,B^2>$ 的生成群，这些操作一定不会影响第一阶段的结果。

### 3、公式解法

以 CFOP 解法为例：

1）数学化每一个魔方公式

2）魔方状态的模式匹配

3）根据魔方状态匹配对应的魔方公式，对魔方矩阵进行操作

4）获得下一阶段的魔方状态，重复前面的操作，纸质魔方被复原

## 四、魔方中的数学

1、n 为 m 阶 魔方的定义

1）n 维立方体的魔方，一共有 2n 个表面，对应 2n 个颜色

​		对应于定理：维立方体n维立方体 中， n−k 维骨架的 维立方体n−k维立方体 的数量是$\left(  C_n^k\right) \times 2^k$

2）只讨论三维魔方，高纬的没有意思，就是抽象代数了

魔方上帝之数的证明：

​	暴力永远管用，除非不够暴力。

暴力验证了 三阶魔方 的所有状态的复原步数都不超过 20



