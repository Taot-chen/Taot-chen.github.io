---
layout: post
title: triton_bug_solve
date: 2025-06-14
tags: [triton]
author: taot
---


### 1 conda 环境 version `GLIBCXX_3.4.30‘ not found

在 conda 虚拟环境中安装好 triton 后，跑 `triton/python/tutorials/01-vector-add.py`的时候出现报错：
```bash
ImportError: /home/taot/anaconda3/envs/triton/bin/../lib/libstdc++.so.6: version `GLIBCXX_3.4.30' not found (required by /home/taot/code/triton-code/triton/python/triton/_C/libtriton.so)
```

**解决方法：**

参考：https://blog.csdn.net/qq_38342510/article/details/140452160?spm=1001.2014.3001.5501

检查是否存在
```bash
strings /usr/lib/x86_64-linux-gnu/libstdc++.so.6 | grep GLIBCXX
```

结果如下，这里是存在version `GLIBCXX_3.4.30`的
```bash
GLIBCXX_3.4
GLIBCXX_3.4.1
GLIBCXX_3.4.2
GLIBCXX_3.4.3
GLIBCXX_3.4.4
GLIBCXX_3.4.5
GLIBCXX_3.4.6
GLIBCXX_3.4.7
GLIBCXX_3.4.8
GLIBCXX_3.4.9
GLIBCXX_3.4.10
GLIBCXX_3.4.11
GLIBCXX_3.4.12
GLIBCXX_3.4.13
GLIBCXX_3.4.14
GLIBCXX_3.4.15
GLIBCXX_3.4.16
GLIBCXX_3.4.17
GLIBCXX_3.4.18
GLIBCXX_3.4.19
GLIBCXX_3.4.20
GLIBCXX_3.4.21
GLIBCXX_3.4.22
GLIBCXX_3.4.23
GLIBCXX_3.4.24
GLIBCXX_3.4.25
GLIBCXX_3.4.26
GLIBCXX_3.4.27
GLIBCXX_3.4.28
GLIBCXX_3.4.29
GLIBCXX_3.4.30
GLIBCXX_DEBUG_MESSAGE_LENGTH
```

建立软链接：
```bash
cd /home/taot/anaconda3/envs/triton/bin/../lib/
mv libstdc++.so.6 libstdc++.so.6.old
ln -s /usr/lib/x86_64-linux-gnu/libstdc++.so.6 libstdc++.so.6
```


### 2 `TORCH_COMPILE_DEBUG=1 python3 torch_triton_kernel.py`报错`TypeError: must be called with a dataclass type or instance`

在某次执行 `TORCH_COMPILE_DEBUG=1 python3 torch_triton_kernel.py` 的时候，遇到报错：
`TypeError: must be called with a dataclass type or instance`
报错的代码如下脚本的第9行`new_model = torch.compile(model, backend="inductor")`:

```python
import torch

def model(input):
    a = torch.add(input, input)
    b = torch.sin(a)
    c = torch.sqrt(b)
    return c

new_model = torch.compile(model, backend="inductor")

input = torch.rand((333, 444, 555), dtype=torch.float16)
output = new_model(input)
```


**解决方法：**

参考 issue：https://github.com/triton-lang/triton/issues/5026

这个报错的原因是 torch 的版本和 triton 的版本不配套，需要修改 torch 的版本或者 triton 的版本。

卸载 torch 之后重新安装 torch 2.3.0，此时会自动卸载源码编译的 triton，并安装 triton 2.3.0，此时这个报错即可解决。如果需要自己使用源码编译安装 triton，那么就需要把 triton的版本切到 2.3.x。之后再重新编译安装 triton 即可。
