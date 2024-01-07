---
layout: post
title: python_program_multifile
date: 2023-12-24
tags: [python]
author: taot
---

## Python多文件编程

### **1、源代码放在同一文件夹**

**_例如：_** 在同一个文件夹中，新建两个文件：test.py, main.py

&emsp; 在 test.py 里面定义一个 class 类，然后在 main.py 里面调用 test.py 里面定义的方法和类：

```python
#test.py文件
        class student:
            def __init__(self,name,age):
                self.name=name
                self.age=age
            def print1(self):
                print("我的名字是%s,我的年龄是%d"%(self.name,self.age))


     # main1.py文件
        import test      #首行必须要先导入test,即文件名
        a = test.student("李华",18) #调用test文件里面的方法和类，前面必须要先加上那个文件名
        a.print1()
```

然后用命令 python main1.py 或者在集成开发环境中运行 main.py 就可以看到程序输出的结果。

### **2、源代码不在同一个文件夹中**

必须要先导入那个类，即找到那个文件的路径

&emsp; 还是以上面两个文件为例，假设其中 test.py 在一个文件夹里面，mian.py 在另一个文件夹里面。

```python
 # main.py 文件
import sys
sys.path.append("test.py的绝对路径")  #导入test文件的绝对路径
import test
a= test.student("李华",18)
a.print1()

```

然后用命令 python main1.py 或者在集成开发环境中运行 main.py 就可以看到程序输出的结果。

