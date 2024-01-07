---
layout: post
title: markdown_math_formulas
date: 2023-12-24
tags: [tools]
author: taot
---

## MarkDown数学公式基本语法

### 1、公式排版

​		MarkDown中公式公式的语法与`LaTeX`类似，排版可以分为两种：

* **行内公式：** 使用`$`包裹公式

**例如：**

`$E=mc^2$` --> $E=mc^2$

* **独立公式：** 使用`$$`包裹公式

**例如：**

```markdown
$$
E=mc^2
$$
```

显示为：
$$
E=mc^2
$$

*  **`\boxed`命令给公式加一个边框**  ，例如：

```markdown
$$
\boxed{E=mc^2}
$$
```

$$
\boxed{E=mc^2}
$$

### 2、特殊转义字符

​		`# $ & ~ _ ^ \ { }`这些字符在`MarkDown`中有特殊的意义，在需要使用这些字符的时候，需要进行转义：

`\#` --> \# $\quad$ `\$` --> \$ $\quad$ `\&` --> \&

`\~` --> \~ $\quad$ `\_` --> \_ $\quad$ `\^` --> \^

`\\` --> \\ $\quad$  `\{` --> \{ $\quad$  `\}` --> \}

### 3、希腊字母

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



### 4、上下标

上下标分别使用`^`和`_`表示。例如：

`$x^2$` --> $x^2$

`$x_2$` --> $x_2$

* 默认情况下，**上下标符号仅仅对下一个组起作用**。一个组即单个字符或者使用`{}`包裹起来的内容。例如：

`$10^10$`会得到 $10^10$ ，要得到 $10^{10}$ ，应该写成`$10^{10}$`。

* 大括号还能消除二义性，如`$x^5^6$`会显示错误，必须使用大括号来界定`^`的结合性，如`${x^5}^6$`：${x^5}^6$，或者`$x^{5^6}$`：$x^{5^6}$。
* 注意区分`$x_i^2$`：$x_i^2$ 和`$x_{i^2}$`：$x_{i^2}$。


### 5、根号、分数、括号、矢量

1）根号：通用表达方式为`$\sqrt[a]{b}$` --> $\sqrt[a]{b}$

* `[]`内的`a`表示开`a`次方，若省略则表示开平方，`$\sqrt{b}$` --> $\sqrt{b}$
* 如果被开方的是单个字符，`{}`可以省略，`$\sqrt[a]b$` --> $\sqrt[a]b$

2）分式：分式有两种表示方法

* 第一种使用`$\frac {a}{b}$` --> $\frac {a}{b}$。当`a`和`b`是单个字符时，可以省略`{}`。
* 第二种使用`$\over$`来分割一个组的前后两部分，`$a+1 \over b+1$` --> $a+1 \over b+1$。

3）括号

* **小括号和方括号：** 使用原始的`()`和`[]`即可。`$(2+3)[4+4]$` --> $(2+3)[4+4]$。
* **大括号：** 由于大括号`{}`被用来分组，因此需要使用`\{`和`\}`来进行转义表示大括号，也可以使用`\lbrace`和`\rbrace`来表示。如`$\{a*b\}$`或者`$\lbrace a*b \rbrace$`，都会显示为 $\lbrace a*b \rbrace$。
* **尖括号：** 使用`\langle`和`\rangle`分别表示左尖括号和右尖括号。`$\langle x \rangle$` --> $\langle x \rangle$。
* **向上取整：** 使用`\lceil`和`\rceil`表示。`$\lceil x \rceil$` --> $\lceil x \rceil$。
* **向下取整：** 使用`\lfloor`和`\rfloor`表示。`$\lfloor x \rfloor$` --> $\lfloor x \rfloor$。

*注：* 原始括号不会随公式大小缩放。例如`$(\frac 12)$` --> $(\frac 12)$。使用`\left( ...\right)`可以自适应地调整括号。例如`$\left( \frac 12 \right)$` --> $\left( \frac 12 \right)$。

### 6、数学运算符与数学符号

1）常规使用`+ - * / =`这五个直接输入即可。

2）特殊形式的数学运算符与数学符号如下表：

|       符号       |         TeX          |        符号        |           TeX           |        符号         |          TeX          |
| :--------------: | :------------------: | :----------------: | :---------------------: | :-----------------: | :-------------------: |
|      $\pm$       |       `$\pm$`        |       $\mp$        |         `$\mp$`         |       $\cdot$       |       `$\cdot$`       |
|     $\times$     |      `$\times$`      |       $\div$       |        `$\div$`         |       $\star$       |       `$\star$`       |
|      $\ast$      |       `$\ast$`       |       $\cup$       |        `$\cup$`         |       $\cap$        |       `$\cap$`        |
|      $\lor$      | `$\vee$`或者`$\lor$` |      $\wedge$      | `$\wedge$`或者`$\land$` |      $\simeq$       |      `$\simeq$`       |
|     $\oplus$     |      `$\oplus$`      |     $\otimes$      |       `$\otimes$`       |       $\sim$        |       `$\sim$`        |
|     $\circ$      |      `$\circ$`       |     $\bullet$      |       `$\bullet$`       |      $\subset$      |      `$\subset$`      |
| $\bigtriangleup$ |  `$\bigtriangleup$`  | $\bigtriangledown$ |   `$bigtriangledown$`   |      $\supset$      |      `$\supset$`      |
|     $\nabla$     |      `$\nabla$`      |     $\exists$      |       `$\exists$`       |     $\subseteq$     |     `$\subseteq$`     |
|    $\partial$    |     `$\partial$`     |      $\infty$      |       `$\infty$`        |     $\supseteq$     |     `$\supseteq$`     |
|    $\forall$     |     `$\forall$`      |      $\surd$       |        `$\surd$`        |        $\in$        |        `$\in$`        |
|     $\angle$     |       `\angle`       |       $\bot$       |        `$\bot$`         |        $\ni$        | `$\ni$`或者`$\owns$`  |
|      $\leq$      | `$\leq$`或者`$\le$`  |       $\geq$       |   `$\geq$`或者`$\ge$`   |      $\notin$       |      `$\notin$`       |
|     $\equiv$     |      `$\equiv$`      |     $\approx$      |       `$\approx$`       |       $\neq$        |  `$\neq$`或者`$\ne$`  |
|      $\lll$      |       `$\lll$`       |       $\ggg$       |        `$\ggg$`         |       $\cong$       |       `$\cong$`       |
|    $\propto$     |     `$\propto$`      |  $\varsubsetneqq$  |   `$\varsubsetneqq$`    |  $\varsupsetneqq$   |  `$\varsupsetneqq$`   |
|      $\mid$      |       `$\mid$`       |   $\Rrightarrow$   |    `$\Rrightarrow$`     |    $\Lleftarrow$    |    `$\Lleftarrow$`    |
|   $\parallel$    |    `$\parallel$`     | $\upharpoonright$  |   `$\upharpoonright$`   | $\downharpoonright$ | `$\downharpoonright$` |
|    $\because$    |     `$\because$`     |    $\therefore$    |     `$\therefore$`      |                     |                       |

### 7、注音与标注

`$\bar{x}$` --> $\bar{x}$ $\quad$ `$\acute{x}$` --> $\acute{x}$ $\quad$  `$\check{x}$` --> $\check{x}$ $\quad$ `$\grave{x}$` --> $\grave{x}$

`$\vec{x}$` --> $\vec{x}$ $\quad$   `$\hat{x}$` --> $\hat{x}$ $\quad$  `$\tilde{x}$` --> $\tilde{x}$ $\quad$ `$\breve{x}$` --> $\breve{x}$

`$\dot{x}$` --> $\dot{x}$ $\quad$`$\ddot{x}$` --> $\ddot{x}$ $\quad$   $\quad$ `$\mathring{x}$` --> $\mathring{x}$

`$\overline{xxx}$` --> $\overline{xxx}$ $\quad$`$\overleftrightarrow{xxx}$` -->  $\overleftrightarrow{xxx}$

`$\underline{xxx}$` --> $\underline{xxx}$ $\quad$`$\underleftrightarrow{xxx}$` -->  $\underleftrightarrow{xxx}$

`$\overleftarrow{xxx}$` --> $\overleftarrow{xxx}$ $\quad$`$\overbrace{xxx}$` -->  $\overbrace{xxx}$

`$\underleftarrow{xxx}$` --> $\underleftarrow{xxx}$ $\quad$`$\underbrace{xxx}$` -->  $\underbrace{xxx}$

`$\overrightarrow{xxx}$` --> $\overrightarrow{xxx}$ $\quad$`$\widehat{xxx}$` -->  $\widehat{xxx}$

`$\underrightarrow{xxx}$` --> $\underrightarrow{xxx}$ $\quad$`$\widetilde{xxx}$` -->  $\widetilde{xxx}$

### 8、省略号、空白间隔、分界符

1）省略号：省略号用 `\dots \cdots \vdots \ddots`表示。

* `$\dots$` --> $\dots$，位置比较低，一般用于有下标的序列：

```markdown
$$
x_1, x_2, \dots, x_n
$$$
```

$$
x_1, x_2, \dots, x_n
$$

* `$\cdots$` --> $\cdots$，位置居中，一般用于正常序列：

```markdown
$$
1, 2, \cdots, n
$$$
```

$$
1, 2, \cdots, n
$$

* `$\vdots$` --> $\vdots$，竖直省略号，一般用于矩阵中。
* `$\ddots$` --> $\ddots$，$45^o$ 方向省略号，一般用于矩阵中。

2）空白间隔：`$\quad$` --> $\quad$ (1em)

```markdown
$\,$ 3/18em   
$\:$  4/18em  
$\;$ 5/18em 
$\quad$ 1em 
$\qquad$ 2m 
$\!$ -3/18em
```


### 9、字体

1）使用`\it`显示意大利体（公式默认字体）：

`$\it{ACDEFGHIJKLMnopqrstuvwxyzACDEFGHIJKLMnopqrstuvwxyz}$` -->

$\it{ACDEFGHIJKLMnopqrstuvwxyzACDEFGHIJKLMnopqrstuvwxyz}$

2）使用`\mathbb`或`\Bbb`显示黑板粗体（黑板黑体）:

`$\mathbb{CHNQRZ}$` --> $\mathbb{CHNQRZ}$

3）使用`\mathbf`或`\bf`显示黑体：

`$\mathbf{ABCDEFGHIJKLMnopqrstuvwxyzABCDEFGHIJKLMnopqrstuvwxyz}$` -->

$\mathbf{ABCDEFGHIJKLMnopqrstuvwxyzABCDEFGHIJKLMnopqrstuvwxyz}$

4）使用`\mathtt`或`\tt`显示打印机字体：

`$\mathtt{ABCDEFGHIJKLMnopqrstuvwxyzABCDEFGHIJKLMnopqrstuvwxyz}$` -->

$\mathtt{ABCDEFGHIJKLMnopqrstuvwxyzABCDEFGHIJKLMnopqrstuvwxyz}$

### 10、分段函数

`&`表示对齐，`\\`用来表示换行，`\qquad`可以表示空格。

```markdown
$$
函数名=\begin{cases}  
公式1 & 条件1 \\
公式2 & 条件2 \\
公式3 & 条件3 
\end{cases}
$$
```

$$
函数名=\begin{cases}  
公式1 & 条件1 \\
公式2 & 条件2 \\
公式3 & 条件3 
\end{cases}
$$

### 11、大型数学运算符

|   运算符    |      TeX      |    运算符    |      TeX       |
| :---------: | :-----------: | :----------: | :------------: |
|   $\sum$    |   `$\sum$`    |    $\int$    |    `$\int$`    |
|   $\prod$   |   `$\prod$`   |   $\iint$    |   `$\iint$`    |
|  $\coprod$  |  `$\coprod$`  |   $\iiint$   |   `$\iiint$`   |
|  $\bigvee$  |  `$\bigvee$`  | $\bigwedge$  | `$\bigwedge$`  |
| $\bigoplus$ | `$\bigoplus$` | $\bigotimes$ | `$\bigotimes$` |
|  $\bigcup$  |  `$\bigcup$`  |    $\lim$    |    `$\lim$`    |

1）使用上标和下标分别表示运算分的上下限：

`$\sum_0^\infty$` --> $\sum_0^\infty$ $\quad$ `$\int_{-\infty}^{\infty}$` -->$\int_{-\infty}^{\infty}$ $\quad$` $\lim_{x\to0} \frac {sinx}x$` -->  $\lim_{x\to0} \frac {sinx}x$

2）使用`\to`表示**趋近于**的箭头：`$x\to0$` -->$x\to0$

3）和、积、极限、积分等运算符用`\sum, \prod, \lim, \int,`这些公式在行内公式被压缩，以适应行高，可以通过`\limits`和`\nolimits`命令显示制动是否压缩。

`$\int\limits_{-\infty}^{\infty} \frac {sinx}xdx$` --> $\int\limits_{-\infty}^{\infty} \frac {sinx}xdx$

`$\int\nolimits_{-\infty}^{\infty} \frac {sinx}xdx$` --> $\int\nolimits_{-\infty}^{\infty} \frac {sinx}xdx$

`$\lim \limits_{n \to +\infty} \frac{n-1}{n(n+1)(n+2)}$` --> $\lim \limits_{n \rightarrow +\infty} \frac{n-1}{n(n+1)(n+2)}$

`$\lim \nolimits_{n \to +\infty} \frac{n-1}{n(n+1)(n+2)}$` --> $\lim \nolimits_{n \rightarrow +\infty} \frac{n-1}{n(n+1)(n+2)}$



### 12、箭头

`$\leftarrow$` --> $\leftarrow$ $\quad$ `$\rightarrow$` --> $\rightarrow$ $\quad$ `$\Leftarrow$` --> $\Leftarrow$ $\quad$ `$\Rightarrow$` --> $\Rightarrow$

`$\leftrightarrow$` --> $\leftrightarrow$ $\quad$ `$\Leftrightarrow$` --> $\Leftrightarrow$ $\quad$ 

`$\longleftarrow$` --> $\longleftarrow$ $\quad$ `$\longrightarrow$` --> $\longrightarrow$ $\quad$ `$\Longleftarrow$` --> $\Longleftarrow$ $\quad$ 

`$\longleftrightarrow$` --> $\longleftrightarrow$ $\quad$ `$\Longleftrightarrow$` --> $\Longleftrightarrow$ $\quad$ `$\Longrightarrow$` --> $\Longrightarrow$

`$\xrightarrow$`和`$\xleftarrow$`可以根据内容自动调整：

```markdown
$$
 \xleftarrow{x+y+z} \quad \xrightarrow[x<y]{x+y+z} 
$$
```

$$
\xleftarrow{x+y+z} \quad \xrightarrow[x<y]{x+y+z}
$$

### 13、多行公式

1）长公式：无需对齐的长公式可使用`multline`；需要对齐使用`split`。使用`\\和&`来分行和设置对齐的位置

```markdown
$$
\begin{multline}
	x = a+b+c+{} \\
   		 d+e+f+g
  \end{multline}
$$
```


```markdown
$$
\begin{split}
x = {} & a + b + c +{}\\
	&d + e + f + g
\end{split}
$$
```

$$
\begin{split}
x = {} & a + b + c +{}\\
	&d + e + f + g
\end{split}
$$

2）方程组：不需要对齐的方程组用`gather`，需要对齐使用`align`:

```markdown
$$
\begin{gather}
a = b+c+d\\
x=y+z
\end{gather}
$$
```

$$
\begin{gather}
a = b+c+d\\
x=y+z
\end{gather}
$$

```markdown
$$
\begin{align}
a &=b+c+d \\
x &=y+z
\end{align}
$$
```

$$
\begin{align}
a &=b+c+d \\
x &=y+z
\end{align}
$$

