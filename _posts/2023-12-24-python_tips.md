---
layout: post
title: python_tips
date: 2023-12-24
tags: [python]
author: taot
---

# **Python 的一些小技巧总结**

---

## **一、Python 多文件编程**

### **1、源代码放在同一文件夹**

**_例如：_** 在同一个文件夹中，新建两个文件：test.py,main.py<br>
&emsp; 在 test.py 里面定义一个 class 类，然后在 main.py 里面调用 test.py 里面定义的方法和类：<br>

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

然后用命令 python main1.py 或者在集成开发环境中运行 main.py 就可以看到程序输出的结果。<br>

### **2、源代码不在同一个文件夹中**

必须要先导入那个类，即找到那个文件的路径<br>
&emsp; 还是以上面两个文件为例，假设其中 test.py 在 python 的文件夹里面，mian.py 在另一个文件夹里面。

```python
 # main.py 文件
import sys
sys.path.append("test.py的绝对路径")  #导入test文件的绝对路径
import test
a= test.student("李华",18)
a.print1()

```

然后用命令 python main1.py 或者在集成开发环境中运行 main.py 就可以看到程序输出的结果。<br>

## **二、Python 中关于变量赋值的一个小坑**

&emsp;在使用变量赋值复制的时候，python 的解释器默认的是浅复制（shadow copy）。这会在需要对该变量进行临时修改的时侯埋下暗雷：再次需要使用该变量的值的时候，变量的值会随着临时变量的修改而被修改。

```python
import numpy as np
a=np.linspace(0,1,100)
b=a
b+=0.1
print(a)
print(b)
```

## **三、a=a+1 和 a+=1 的区别**

&emsp;首先 python 中变量名是变量的标签，因此重新赋值在某种意义上是标签的重新指向，而在 a += 1 和 a = a + 1 两个问题上，更准确是说是对于不可变对象类型，是重新开辟空间，这个标签指向新的内存空间，对于可变对象类型，则是在原来内存上做操作。

简言之：<br>

- 对于可变对象，a += 1 直接在原内存地址上操作 a = a + 1 开辟新的内存操作
- 对于不可变对象，a += 1 和 a = a + 1 都是在新开辟的空间操作
    **_例程见二中的例子_**

## **四、Times New Roman字体突然找不到**

&emsp;原本正常运行的程序，突然运行不正常，并且提示找不到新罗马字体（程序中在画图的时候，设置图像样式使用了新罗马字体）。
&emsp;不正常体现在运行到 **plt.show()** 的时候，程序就会自己停止运行，图片也不会显示出来。

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

即可。

## **五、获取路径**

1）获取当前文件所在目录的路径：

```python
import os
dir_path=os.path.dirname(__file__)
```

2）获取当前文件路径：

```python
import os
# 当前文件路径
file_path=os.getcwd()
file_path=__file__

#当前文件的父路径
father_path=os.path.abspath(os.path.dirname(pwd)+os.path.sep+".")

#当前文件的前两级目录
grader_father=os.path.abspath(os.path.dirname(pwd)+os.path.sep+"..")
```

3）遍历文件夹中的所有文件：

```python
import os
 
path = '待访问文件夹路径'
path_list = os.listdir(path)
path_list.remove('.DS_Store')    # macos中的文件管理文件，默认隐藏，这里可以忽略
path_list.sort(key=lambda x:int(x.split('.')[0]))   # 对文件名进行排序
print(path_list)
for filename in path_list:
    f = open(os.path.join(path,filename),'rb')
```

3）遍历文件夹中的所有文件,并对这些文件夹的名称按照一定的规则排序：

```python
# 例如在当前程序所在的文件夹内有一系列的文件夹，是以下划线加数字的形式命名的。现在需要获取当前文件夹中所有子文件夹的名称，并把这些文件夹的名称按照其中的数字来排序，方便后续的使用
# 例如这些文件夹的名称都是如下：_123.123,_234.2321,_213.214等形式
# 获取到这些文件夹的名称后，将其存放在列表中，并按照数字从大到小降序排列

import os
import functools

# 获取当前文件所在的路径
path=os.path.dirname(__file__)

# sorted函数的cmp函数
def cmp(a, b):
    a = a[1:]
    b = b[1:]
    num_a = float(a)
    num_b = float(b)
    if num_a > num_b:
        return -1
    elif num_a < num_b:
        return 1
    else:
        return 0


# 获取给定路径下的所有子文件夹的名称
# 并排序
def dirListGet(targetPath):
    nameList = os.listdir(targetPath)
    # 去掉nameList里面的非文件夹名称
    dirNameList = []
    for item in nameList:
        if item[-3:] != ".py" and item[-4:] != ".txt" and item[-4:] != ".png":
            dirNameList.append(item)

    # 对dirNameList的元素进行排序
    sortedDirNameList = sorted(dirNameList, key=functools.cmp_to_key(cmp))

    return sortedDirNameList

dirList=dirListGet(path)



```

## 六、函数拟合次数超过maxfev=1000

&emsp;在对实验数据进行双指数拟合的时候，有时候会出现如下报错：

```python
Optimal parameters not found: Number of calls to function has reached maxfev = 1000
```

这是因为在python里面函数拟合的次数上限由参数 amxfev 控制。解决方式是在函数拟合的语句内修改参数 maxfev 的默认值：

```python
popt, pcov = curve_fit(double_exp, x, y, [1, 1, 1, 1], maxfev=500000)
```

## 七、获取二维数组的某一列、某一行

```python
# a是一个二维数组
# 获取a的第一列
b=a[:,0]
# 获取a的第一行
b=a[0,:]	或者	b=a[0]

# b是一个二维列表
# 从a中的每一行取第一个元素
b = [i[0] for i in a] 

```

