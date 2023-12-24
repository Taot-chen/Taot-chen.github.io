---
layout: post
title: python_plot_with_new_timesroman
date: 2023-12-24
---

## Python绘图时使用Times New Roman字体

* 在画图的时候，设置图像样式时使用新罗马字体，报错提示找不到Times New Roman字体
* 系统环境：ubuntu20.04LTS
* Python版本：python3.8.5

**解决方法**
在代码中加入：

```python
import matplotlib; matplotlib.use('TkAgg')
```

会提示"No module named _tkinter"；
再在*Teiminal* 中安装 *python3-tk* 类库:

```shell
sudo apt install python3-tk
```

此时运行程序，即可正常运行。
若依然报错，可安装 *tk* 的开发类库:

```shell
sudo apt install tk-dev
```

至此，一般会得到解决。

