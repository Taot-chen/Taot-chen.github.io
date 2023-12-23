---
layout: post
title: python_languge_features
date: 2023-12-23
---

# python languge features

## 1、python argparse bool 解析
* python 的 argparse，对于输入全部都是按照字符串读取，即便设置了 type=bool，依然是按照字符串获取。因此不论该参数设什么值（True/False），只要设置了，都会被获取成非空字符串，后续转 bool 就都是 true 了
* 可以额外使用 action 参数，或者是添加一个 str2bool 的方法，来处理 boolean 的命令行参数
  * action 参数，action关键字默认状态有两种，store_true和store_false
    * 输入命令时，不指定其参数，则store_true显示为False,store_false显示为True
    ```pyhton
        parse.add_argument("--a", action="store_true")
    ```
    action类型参数 a 的值为 store_true，若命令行输入时不指定 a, 那么结果默认为 False，在命令行输入了 `--a` 参数，那么结果为 True
  * 添加一个 str2bool 的方法，来处理 boolean 的命令行参数
    ```python
        parse.add_argument("--a", default=False, type=str2bool)
    ```
    str2bool
    ```python
        def str2bool(value):
          if isinstance(value, bool):
              return value
          if value.lower() in ("yes", "true", "t", "y", "1"):
              return True
          elif value.lower() in ("no", "false", "f", "n", "0"):
              return False
          else:
              raise argparse.ArgumentTypeError("Boolean value expected")
    ```

## 2、python 禁用 print 输出
有时候在调试代码的时候会打印一些 log，但是调试完又懒得一一去掉，可以通过控制`sys.stdout`来实现`print`输出的开关
```python
    import sys
    print("aha")

    #关闭print的输出
    sys.stdout = open(os.devnull, 'w')
    print("ohuo")

    #打开print的输出
    sys.stdout = sys.__stdout__
    print("aha")

```