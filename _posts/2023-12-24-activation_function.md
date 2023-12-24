---
layout: post
title: activation_function
date: 2023-12-24
---

## 激活函数

* 激活函数，决定神经网络是否传递信息的开关

    * ReLU，Recitified Linear Unit，线性整流函数，常见的是 ReLU 和 Leaky ReLU。通常意义下，线性整流函数指代数学中的斜坡函数
        $$
        f(x) = \max (0, x)
        $$
        ReLU  可以对抗梯度爆炸 / 消失的问题，相对而言计算效率也很高

    * GELU，Gaussian Error Linear Unit，高斯误差线性单元

        * 对于输入值 x，根据 x 的情况，乘上 1  或者 0，即对于每一个输入 x，服从标准正态分布 $N(0, 1)$，再给其乘上一个伯努利分布 $\phi(x) = P(X \leq x)$ ：
            $$
            xP(X \leq x) = x \phi(x)
            $$
            其中 $\phi(x)$ 是 $x$ 的高斯分布;
            $$
            xP(X \leq x) = x \int \nolimits _{-\infty} ^{x} \frac{e^{-\frac{(X - \mu)^2}{2 \sigma^2}}}{\sqrt{2 \pi \sigma}}dX
            $$
            $\rightarrow$ 
            $$
            gelu(x) = 0.5x(1+\tanh(\sqrt{\frac{2}{\pi}}(x+0.044715x^3)))
            $$

    * 当 $x$ 越大的时候，就越有可能被保留，越小就越有可能被置零

    * relu，$relu(x) = \max(x, 0)$

    * sigmoid, $sigmoid(x) = \frac{1}{1+e^{-x}}$

    * tanh
        $$
        \sinh(x) = \frac{e^x - e^{-x}}{2} \\
        \cosh(x) = \frac{e^x + e^{-x}}{2} \\
        \tanh(x) = \sinh(x)\cosh(x)
        $$

    * silu, $silu(x) = x * sigmoid(x) = \frac{x}{1+e^{-x}}$

    * gelu
        $$
        gelu(x) \approx 0.5x(1+\tanh(\sqrt{\frac{2}{\pi}}(x+0.044715x^3))) \\
        		\approx x \times sigmoid(1.702x)
        $$

    * mish, $mish(x) = x \times \tanh(softplue(x)) = x \times \tanh(\ln(a + e^x))$

    激活函数近似是往负无穷大方向走，逐渐趋近 $y = a$ 的直线；往正无穷大的方向走，逐渐趋近 $y = x$

    