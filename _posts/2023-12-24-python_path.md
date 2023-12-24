---
layout: post
title: python_path
date: 2023-12-24
---


## Python 路径操作

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

## 