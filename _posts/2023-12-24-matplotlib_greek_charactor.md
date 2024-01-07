---
layout: post
title: matplotlib_greek_charactor
date: 2023-12-24
tags: [python]
author: taot
---

## Matplotlib绘图时希腊字母的显示

### 1、输入格式

* **`Matplotlib`支持`LaTeX`语法，输入格式与`LaTeX`中公式的输入一致。**

例如，在图像（0,0）处显示 $2\mu m$ 可以设置为：

```python
plt.text(0, 0, "2$\mu$m", fontsize=15)
```

* 在包含希腊字母的字符串中，可以同时使用`LaTeX`语法和`.format()`进行字符串拼接。

```python
plt.plot(x[i], y[i], label="{}$\mu$m".format(file_name[i]))
```

### 2、`LaTeX`中常用的希腊字母的写法

|  名称   |    大写    |     TeX      |    小写    |     TeX      |
| :-----: | :--------: | :----------: | :--------: | :----------: |
|  alpha  |    $A$     |    `$A$`     |  $\alpha$  |  `$\alpha$`  |
|  beta   |    $B$     |    `$B$`     |  $\beta$   |  `$\beta$`   |
|  gamma  |  $\Gamma$  |  `$\Gamma$`  |  $\gamma$  |  `$\gamma$`  |
|  delta  |  $\Delta$  |  `$\Delta$`  |  $\delta$  |  `$\delta$`  |
| epsilon |    $E$     |    `$E$`     | $\epsilon$ | `$\epsilon$` |
|  zeta   |    $Z$     |    `$Z$`     |  $\zeta$   |   $\zeta$    |
|   eta   |    $H$     |    `$H$`     |   $\eta$   |   `$\eta$`   |
|  theta  |  $\Theta$  |  `$\Theta$`  |  $\theta$  |   $\theta$   |
|  iota   |    $I$     |    `$I$`     |  $\iota$   |  `$\iota$`   |
|  kappa  |    $K$     |    `$K$`     |  $\kappa$  |  `$\kappa$`  |
| lambda  | $\Lambda$  | `$\Lambda$`  | $\lambda$  | `$\lambda$`  |
|   mu    |    $M$     |    `$M$`     |   $\mu$    |   `$\mu$`    |
|   nu    |    $N$     |    `$N$`     |   $\nu$    |   `$\nu$`    |
|   xi    |   $\Xi$    |   `$\Xi$`    |   $\xi$    |   `$\xi$`    |
| omicron |    $O$     |    `$O$`     | $\omicron$ | `$\omicron$` |
|   pi    |   $\Pi$    |   `$\Pi$`    |   $\pi$    |   `$\pi$`    |
|   rho   |    $P$     |    `$P$`     |   $\rho$   |   `$\rho$`   |
|  sigma  |  $\Sigma$  |  `$\Sigma$`  |  $\sigma$  |  `$\sigma$`  |
|   tau   |    $T$     |    `$T$`     |   $\tau$   |   `$\tau$`   |
| upsilon | $\Upsilon$ | `$\Upsilon$` | $\upsilon$ | `$\upsilon$` |
|   phi   |   $\Phi$   |   `$\Phi$`   |   $\phi$   |   `$\phi$`   |
|   chi   |    $X$     |    `$X$`     |   $\chi$   |   `$\chi$`   |
|   psi   |   $\Psi$   |   `$\Psi$`   |   $\psi$   |   `$\psi$`   |
|  omega  |  $\Omega$  |  `$\Omega$`  |  $\omega$  |  `$\omega$`  |

