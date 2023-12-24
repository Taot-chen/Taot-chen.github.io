---
layout: post
title: polynomial_fit
date: 2023-12-23
---


## 多项式拟合算法

### 1、多项式拟合

* 多项式拟合 & 多项式插值

    * 多项式插值，给出一些点，求出这些点所满足的方程的表达式，表达式曲线一定经过每个点

        $\boxed{n 次多项式插值} \xrightarrow{n+1个点} \boxed{求解 n+1 元一次方程组}$

    * 多项式拟合，给出一些点，求出与这些点最接近的曲线表达式，表达式不一定经过每个点

        $\boxed{n 次多项式拟合} \xrightarrow[最小二乘法]{m 个点，m \ge n+1} \boxed{求解 n+1 元一次方程组}$

* 最小二乘法曲线拟合

    * 最小二乘法是通过最小化残差平方来寻找数据的最优函数匹配

	假设样本数据集 $P(x, y)$ 包含数据点 $P_i(x_i, y_i), (i=1, 2, ..., m)$，来自多项式函数 $f(x) = a_0 + a_1x + a_2x^2 + ... + a_nx^n$ 的多次采样，那么就可以得到方程组：
        $$
        a_0 + a_1x_i + a_2x_i^2 + ... + a_nx_i^n = y_i, (i = 1, 2, ..,, m, m \ge n + 1)
        $$
        用矩阵可以表示为：
        $$
        \begin{bmatrix}
        1 & {x_0} & {x_0^2} & {\cdots} & {x_0^n} \\
        1 & {x_1} & {x_1^2} & {\cdots} & {x_1^n} \\
        {\vdots} & {\vdots} & {\ddots} & {\vdots} \\
        1 & {x_m} & {x_m^2} & {\cdots} & {x_m^n}
        \end{bmatrix}
        \begin{bmatrix}
        {a_0} \\
        {a_1} \\
        {\vdots} \\
        {a_m}
        \end{bmatrix}
        =
        \begin{bmatrix}
        {y_0} \\
        {y_1} \\
        {\vdots} \\
        {y_n}
        \end{bmatrix}
        $$
        $\Rightarrow Xa = Y$ 

    那么样本数据集的残差平方和可以表示为：
        $$
        E = (Xa - Y)^T(Xa - Y)
        $$
        $E$ 对 $a$ 求导得到：
        $$
        \frac{\partial E}{\partial a} = 2X^TXa - 2X^TY
        $$
        显然，$E$ 是关于 $a$  的二次函数，那么在 $\frac{\partial E}{\partial a} = 0$ 时，$E$取到最小值解得：
        $$
        a = (X^TX)^{-1}X^TY
        $$
        $\Rightarrow$ 可以直接计算 $(X^TX)^{-1}X^TY$，求得拟合参数 $a$，

    **这种方式有些问题，在求解逆矩阵的过程中可能后出现奇异值，导致误差比较大，需要自定义大数类**

    * 继续观察方程组矩阵
        $$
        Xa = Y \\
        \rightarrow (X^TX)^{-1}X^TXa = (X^TX)^{-1}X^TY \\
        \rightarrow a = (X^TX)^{-1}X^TY
        $$
        考虑到：
        $$
        Xa = Y \\
        \rightarrow X^{-1}Xa = X^{-1}Y \\
        \rightarrow a = X^{-1}Y
        $$
        由于：

        *   对矩阵左乘初等矩阵就相当于对矩阵做一次初等行变换；
        *   n 阶矩阵 A 可逆的充要条件是 A 可以表示为一系列初等矩阵的乘积

        $\rightarrow$ 那么使用高斯消元也可以得到等价的结果，**这种方式不需要求逆矩阵，可以只使用初等行变换来实现，不易引入较大的误差**

    * 根据这样的方式，实现的多项式拟合类，具有比较可观的性能和精度

### 2、不等式约束条件的优化问题

* 等式约束优化问题

    * 消元法：一阶导数等于 0 求驻点，高斯消元

    * 拉格朗日乘子法：多维、高次耦合的非线性约束问题

        * 等式约束最优化问题
            $$
            \begin{cases}
            min f(\bf{X}) & \bf{X} \in E^n \\
            S.t.g_i(\bf{X}) = 0 & i = 1, 2, \cdots,  m
            \end{cases}
            $$
            引入拉格朗日系数，$\lambda _1, \lambda _2, \cdots, \lambda _m$，构造拉格朗日函数：
            $$
            L(\bf{X}, \lambda _1, \lambda _2, \cdots, \lambda _m) = f(\bf{X}) - \sum _{i = 1}^m \lambda _i g_i(\bf{X})
            $$
            $\rightarrow$
            $$
            \begin{cases}
            \frac{\partial L}{\partial \bf{X}} = \frac{\partial f}{\partial \bf{X}} - \lambda ^T \frac{\partial G}{\partial \bf{X}} = 0 & (n 个方程) \\
            \frac{\partial L}{\partial \lambda} = G(\bf{X}) = 0 & (m 个方程)
            \end{cases}
            $$
            $\rightarrow$ (m + n)  个方程，(m + n) 个未知数，可以使用前面的方法求解方程组

* 不等式约束优化问题

    * 对不等式约束条件引入松弛变量，转化为等式约束优化问题
        $$
        \begin{cases}
        min f(\bf{X}) & \bf{X} \in E^n \\
        S.t.g_i(\bf{X}) \leq 0 & i = 1, 2, \cdots, m
        \end{cases}
        $$
        引入松弛变量 $\nu$，把不等式约束条件转化为等式约束条件：

        $$
        g_i(\bf{X}) + \nu _i ^2 = 0,  i = 1, 2, \cdots, m \\
        $$
        
        引入拉格朗日系数 $\lambda _1, \lambda _2, \cdots, \lambda _m$，构造拉格朗日函数：
        $$
        L(\bf{X}, \lambda _1, \lambda _2, \cdots, \lambda _m, \nu _1, \nu _2, \cdots, \nu _m) = f(\bf{X}) - \sum _{i = 1} ^m \lambda _i (g_i(\bf{X}) + \nu _i ^2)
        $$
        $\rightarrow$
        $$
        \begin{cases}
        \frac{\partial L}{\partial x_j} = \frac{\partial f}{\partial x_j} - \sum _{i = 1} ^m \lambda _i \frac{\partial g}{\partial x_j} = 0 & j = 1, 2, \cdots, n \\
        \frac{\partial L}{\partial \lambda _i} = g_i(\bf{X}) + \nu _i ^2 = 0 & i = 1, 2, \cdots, m \\
        \frac{\partial L}{\partial \nu _i} = -2 \lambda _i \nu _i & i = 1, 2, \cdots, m \\
        \end{cases}
        $$
        $\rightarrow$ (2m + n) 个方程，(2m + n) 个未知数，可以使用前面的方法求解方程组

    * 不等式约束优化问题引入松弛变量的作用是为了将不等式约束转化为等式约束，可以这么做的原理：松弛变量是等高线负值，将不等式约束的取值面改为等式约束的取值曲线，但要满足不等式约束条件。这个方法求解得到的解可能不满足原不等式约束条件，需要通过 KKT 条件判定解是否存在，并利用 KKT 条件求解极值

* KKT 条件

    * KKT 条件给出了不等式约束条件的优化问题，存在极值点的必要条件，即通过判定条件来判定解是否存在，并利用 KKT 条件求解极值

    * 定理，对于不等式约束的非线性最优化问题，
        $$
        \begin{cases}
        min f(\bf{X}) & \bf{X} \in E^n \\
        S.t \ g(\bf{X}) \leq 0 & i = 1, 2, \cdots, m \\
        f(\bf{X}), \ g(\bf{X}) 均可微
        \end{cases}
        $$
        极值点存在的必要条件是：
        $$
        \begin{cases}
        \frac{\partial f(\bf{X})}{\partial x_j} + \sum _{i = 1} ^m \lambda _{i} \frac{\partial g_i(\bf{X})}{\partial x_j} = 0 & i = 1, 2, \cdots, m;\ j = 1, 2, \cdots, n \\
        \lambda _i \geq 0 & (\lambda _i, KKT \ factor) \\
        \lambda _i g_i(\bf{X}) = 0 \\
        g_i(\bf{X}) \leq 0 \\
        \end{cases}
        $$

* KKT  条件的简单证明和一些说明

    * 引入松弛变量 $S_i$，使得：
        $$
        S_i + g_i(\bf{X}) = 0,  i = 1, 2, \cdots, m \\
        $$
        引入拉格朗日乘子，构造拉格朗日方程：
        $$
        L(\bf{X}, \bf{\lambda}, \bf{S}) = f(\bf{X}) + \sum _{i=1}^m \lambda _i[S_i + g_i(\bf{X})]
        $$
        那么：

        ​	1）$\frac{\partial L}{\partial x_j} = \frac{\partial f}{\partial x_j} + \sum _{i=1}^m \lambda _i \frac{\partial g}{\partial x_j} = 0$

        ​	2）$\frac{\partial L}{\partial \lambda _i} = S_i + g_i(\bf{X}) = 0$，考虑到 $S_i \geq 0$，那么 $g_i(\bf{X}) \leq 0$

        ​	3）考虑到当前的讨论点是 $L(\bf{X}, \bf{\lambda}, \bf{S})$ 的极小值点，那么：$dL \geq 0 \rightarrow \frac{\partial L}{\partial S_i} dS_i \geq 0 \rightarrow \lambda _i dS _i \geq 0$，那么不难得到：
        $$
        \begin{cases}
        X 在边界时，dS_i \geq 0 \rightarrow \lambda _i \geq 0 \\
        X 在边界内时，dS_i 可正可负 \rightarrow \lambda _i = 0 \\
        \end{cases}
        $$
        ​		$\rightarrow \lambda _i \geq 0$

        ​	4）在边界上时，$g_i(\bf{X}) = 0, \lambda _i \geq 0$；在边界内时，$g_i(\bf{X}) \leq 0, \lambda _i = 0$

        ​		$\rightarrow \lambda _i g_i(\bf{X}) = 0$

    * 一些说明

        *   一般情况下，KKT 条件是判别有约束极值点的必要非充分条件，但是对于凸规划的情况， KKT 是充要条件
        *   构造拉格朗日函数的 KKT 乘子符号规律：
            *   求 $\min f(\bf{X}), g_i(\bf{X}) \leq 0$ 时，$\lambda _i \geq 0$
            *   求 $min f(\bf{X}), g_i(\bf{X}) \geq 0$ 时，$\lambda _i \leq 0$
            *   求 $\max f(\bf{X}), g_i(\bf{X}) \geq 0$ 时，$\lambda _i \geq 0$
            *   求 $\max f(\bf{X}), g_i(\bf{X}) \leq 0$ 时，$\lambda _i \leq 0$
        *   KTT 条件适用于等式约束条件和不等式约束条件，但是只适用于凸规划，不适用于凹规划

* 凸规划，求最优化问题 P：$\min f(x)$，当 D 为 凸集，f(x) 为凸函数，那么该规划为凸规划

    * 凸集，点集 D 中的任意两点的连线都属于 D，那么 D 就是凸集

        *   一维上的凸集是单点或一条不间断的线（包括直线、射线、线段）
        *   二、三维空间中的凸集就是直观上凸的图形（例如，在二维空间中有扇面、圆、椭圆等，在三维中有实心球体等。多数情况下，两个凸集的交集也是凸集，空集也是凸集）

    * 凸函数

        * 一阶条件：D 为非空凸集，f(x) 在 D 上的所有一阶偏导数都连续，那么 f(x) 在 D 上严格凸的充要条件为：
            $$
            \frac{f(x_2) - f(x_1)}{x_2 - x_1} \geq f^{'}(x_1)
            $$
            即，对于 $\forall \bf{x}, \bf{y} \in D$：
            $$
            f(\bf{y}) \geq f(\bf{x}) + (\bf{y} - \bf{x}) ^T \nabla f(\bf{x})
            $$
            $\bf{x} = \bf{y}$ 时，等号成立

        * 二阶条件：D 为非空开凸集，f(x) 在 D 上的所有二阶偏导数都连续，那么：

            *   f(x) 在 D 上为凸函数的充要条件为，对于 $\forall \bf{x} \in D, \nabla ^2 f(\bf{x}) \geq 0$
            *   若对于 $\forall \bf{x} \in D, \nabla ^2 f({\bf{x}}) > 0$，那么 $f(\bf{x})$ 在 D 上严格凸（逆定理不成立）

* 拉格朗日乘子法简单证明 —— 以二维问题为例

    如果极值点存在，则极值点满足，目标函数在约束曲线的切线方向的方向导数为 0。目标函数 $f(x)$ 沿约束曲线 $g(x_1, x_2) = 0$ 的切线 $s$ 方向的方向导数等于 0，即：
    $$
    \frac{\partial f}{\partial s} = \frac{\partial f}{\partial x_1} \times \frac{\partial x_1}{\partial s} + \frac{\partial f}{\partial x_2} \times \frac{\partial x_2}{\partial s} = 0
    $$
    **注：方向导数代表了曲线梯度在曲线切线方向的分量，梯度是曲线在该点上升的方向。如果曲线梯度与曲线切线向量的乘积不为 0 ，即梯度与曲线切线不垂直，则在该曲线上，沿着负梯度方向走，必然能找见更小的点**

    另外，也不难知道，极值点满足约束函数在约束曲线切线方向的导数也为 0：
    $$
    \frac{\partial g}{\partial s} = \frac{\partial g}{\partial x_1} \times \frac{\partial x_1}{\partial s} + \frac{\partial g}{\partial x_2} \times \frac{\partial x_2}{\partial s} = 0
    $$
    联立上面两个函数在约束曲线切线的极值点条件可得：
    $$
    \frac{\partial f}{\partial x_1} / \frac{\partial g}{\partial x_1} = \frac{\partial f}{\partial x_2} / \frac{\partial g}{\partial x_2} = \lambda
    $$
    结合拉格朗日方程和其偏导数等于 0 的方程组：
    $$
    L(x_1, x_2, \lambda) = f(x_1, x_2) - \lambda g(x_1, x_2) \\
    \begin{cases}
    \frac{\partial L}{\partial x_1} = \frac{\partial f}{\partial x_1} - \lambda \frac{\partial g}{\partial x_1} = 0 \\
    \frac{\partial L}{\partial x_2} = \frac{\partial f}{\partial x_2} - \lambda \frac{\partial g}{\partial x_2} = 0 \\
    \frac{\partial L}{\partial \lambda} = -g(x_1, x_2) = 0 \\
    \end{cases}
    $$
    3 个未知数，3 个独立方程，原问题可解

