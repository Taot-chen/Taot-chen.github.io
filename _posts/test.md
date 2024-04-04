$$
\nabla _A f(A) = 
\left[
\begin{matrix}
\frac{\partial f(A)}{\partial A_{11}} & \frac{\partial f(A)}{\partial A_{12}} & \dotsc & \frac{\partial f(A)}{\partial A_{1n}} \\
\frac{\partial f(A)}{\partial A_{21}} & \frac{\partial f(A)}{\partial A_{22}} & \dotsc & \frac{\partial f(A)}{\partial A_{2n}} \\
\vdots & \vdots & \ddots & \vdots \\
\frac{\partial f(A)}{\partial A_{m1}} & \frac{\partial f(A)}{\partial A_{m2}} & \dotsc & \frac{\partial f(A)}{\partial A_{mn}} \\
\end{matrix}
\right]
$$

即

$$
(\nabla_A f(A))_{ij} = \frac{\partial f(A)}{\partial A_{ij}}
$$

同样地，对于输入向量的函数 $f: R^{n \times 1} \rightarrow R$，那么：

$$
\nabla_x f(x) =
\left[
\begin{matrix}
\frac{\partial f(x)}{\partial x_1} \\
\frac{\partial f(x)}{\partial x_2} \\
\vdots \\
\frac{\partial f(x)}{\partial x_n} \\
\end{matrix}
\right]
$$

根据上面的矩阵和向量的梯度矩阵的定义，可以得到：

$$
\nabla_x (f(x) + g(x)) = \nabla_x f(x) + \nabla_x g(x) \\
\nabla_x (tf(x)) = t\nabla_x f(x), t \in R
$$

