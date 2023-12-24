---
layout: post
title: nosie_reduction_with_python
date: 2023-12-24
---


## Python 数据降噪处理的四种方法

### 一、均值滤波

#### 1）算法思想

&emsp;给定均值滤波窗口长度，对窗口内数据求均值，作为窗口中心点的数据的值，之后窗口向后滑动1，相邻窗口之间有重叠；边界值不做处理，即两端wid_length//2长度的数据使用原始数据。

#### 2）Python实现

```python
'''
均值滤波降噪：
    函数ava_filter用于单次计算给定窗口长度的均值滤波
    函数denoise用于指定次数调用ava_filter函数，进行降噪处理
'''


def ava_filter(x, filt_length):
    N = len(x)
    res = []
    for i in range(N):
        if i <= filt_length // 2 or i >= N - (filt_length // 2):
            temp = x[i]
        else:
            sum = 0
            for j in range(filt_length):
                sum += x[i - filt_length // 2 + j]
            temp = sum * 1.0 / filt_length
        res.append(temp)
    return res


def denoise(t, x, n, filt_length):
    for i in range(n):
        res = ava_filter(x, filt_length)
        x = res
    return (t, res)
```

### 二、奇异值分解

#### 1）算法思想

&emsp;任意m ∗ n 的矩阵A可以分解为如下形式：<br>
A=U·sigema·V(T)<br>
其中U、V分别是左右奇异矩阵，sigema是对角矩阵，对角线上的元素是A的奇异值从大到小的排列。<br>
&emsp;奇异值表示的是原矩阵在其对应特征向量分量上的权重，奇异值越大，对应的特征向量在原矩阵中的权重越大。<br>
&emsp;如果前k（k<r，r是原矩阵的秩）个奇异值数值较大，说明前k个奇异值对应的信息是原矩阵的主成分。那么可以使前k个奇异值不变，其余奇异值设置成0，再重构原矩阵，实现降噪。<br>

#### 2）Python实现

```python
import numpy as np
# import random
import matplotlib.pyplot as plt
import sys
import os


def denoise(t, x):
    # 1、数据预处理
    res = int(np.sqrt(len(x)))
    xr = x[:res * res]
    delay = t[:res * res]

    # 2、一维数组转换为二维矩阵
    x2list = []
    for i in range(res):
        x2list.append(xr[i * res:i * res + res])
    x2array = np.array(x2list)

    # 3、奇异值分解
    U, S, V = np.linalg.svd(x2array)
    S_list = list(S)
    ## 奇异值求和
    S_sum = sum(S)
    ##奇异值序列归一化
    S_normalization_list = [x / S_sum for x in S_list]

    # 4、画图
    X = []
    for i in range(len(S_normalization_list)):
        X.append(i + 1)

    fig1 = plt.figure().add_subplot(111)
    fig1.plot(X, S_normalization_list)
    fig1.set_xticks(X)
    fig1.set_xlabel('Rank', size=15)
    fig1.set_ylabel('Normalize singular values', size=15)
    plt.show()

    # 5、数据重构
    K = 2  ## 保留的奇异值阶数
    for i in range(len(S_list) - K):
        S_list[i + K] = 0.0

    S_new = np.mat(np.diag(S_list))
    reduceNoiseMat = np.array(U * S_new * V)
    reduceNoiseList = []
    for i in range(len(x2array)):
        for j in range(len(x2array)):
            reduceNoiseList.append(reduceNoiseMat[i][j])

    # 6、返回结果
    return (delay, reduceNoiseList)
```


### 三、小波变换

#### 1）算法思想

&emsp;将信号通过小波变换后，信号产生的小波系数含有信号的重要信息，将信号经小波分解后小波系数较大，噪声的小波系数较小，并且噪声的小波系数要小于信号的小波系数，通过选取一个合适的阀值，大于阀值的小波系数被认为是有信号产生的，应予以保留，小于阀值的则认为是噪声产生的，置为零从而达到去噪的目的。<br>

#### 2）Python实现

```python
#模块调用
import numpy as np
import math
import pywt


#封装成函数
def sgn(num):
    if (num > 0):
        return 1.0
    elif (num == 0):
        return 0.0
    else:
        return -1.0


def wavelet_noising(new_df):
    data = new_df
    data = data.values.T.tolist()  # 将np.ndarray()转为列表
    w = pywt.Wavelet('sym8')
    # [ca3, cd3, cd2, cd1] = pywt.wavedec(data, w, level=3)  # 分解波
    [ca5, cd5, cd4, cd3, cd2, cd1] = pywt.wavedec(data, w, level=5)  # 分解波

    length1 = len(cd1)
    length0 = len(data)

    Cd1 = np.array(cd1)
    abs_cd1 = np.abs(Cd1)
    median_cd1 = np.median(abs_cd1)

    sigma = (1.0 / 0.6745) * median_cd1
    lamda = sigma * math.sqrt(2.0 * math.log(float(length0), math.e))
    usecoeffs = []
    usecoeffs.append(ca5)  # 向列表末尾添加对象

    #软硬阈值折中的方法
    a = 0.5

    for k in range(length1):
        if (abs(cd1[k]) >= lamda):
            cd1[k] = sgn(cd1[k]) * (abs(cd1[k]) - a * lamda)
        else:
            cd1[k] = 0.0

    length2 = len(cd2)
    for k in range(length2):
        if (abs(cd2[k]) >= lamda):
            cd2[k] = sgn(cd2[k]) * (abs(cd2[k]) - a * lamda)
        else:
            cd2[k] = 0.0

    length3 = len(cd3)
    for k in range(length3):
        if (abs(cd3[k]) >= lamda):
            cd3[k] = sgn(cd3[k]) * (abs(cd3[k]) - a * lamda)
        else:
            cd3[k] = 0.0

    length4 = len(cd4)
    for k in range(length4):
        if (abs(cd4[k]) >= lamda):
            cd4[k] = sgn(cd4[k]) * (abs(cd4[k]) - a * lamda)
        else:
            cd4[k] = 0.0

    length5 = len(cd5)
    for k in range(length5):
        if (abs(cd5[k]) >= lamda):
            cd5[k] = sgn(cd5[k]) * (abs(cd5[k]) - a * lamda)
        else:
            cd5[k] = 0.0

    usecoeffs.append(cd5)
    usecoeffs.append(cd4)
    usecoeffs.append(cd3)
    usecoeffs.append(cd2)
    usecoeffs.append(cd1)
    recoeffs = pywt.waverec(usecoeffs, w)
    return recoeffs


def denoise(x, data):
    data_denoising = wavelet_noising(data)  #调用小波去噪函数
    return (x, data_denoising)
```

### 四、改变 bin size

#### 1）算法思想

&emsp;通过改变数据的 bin size,来达到降低噪声的目的。<br>
&emsp;改变 bin size 的时候，会导致数据长度减小，降低数据的分辨率。为了最大限度的较少原数据的有效信息的损失，在改变 bin size 的过程中，被抛弃的数据的信息也会保留在保留下来的数据中，具体实现思路是：在给定 bin size = n 的情况下，将 n 长度的数据取平均值作为该区域中心点的数据的值。之后窗口向后滑动 n ，相邻两个窗口之间不重叠。

#### 2）Python实现

```python
# 修改现有数据的bin：
# 即bin=3时：每三个数据，只取中间的一个数据，且这个数据的值为三个数据的平均值
# 在对纵轴进行如上处理的时候，横轴也进行相应的抽值处理：
# 第一个数据不要，第二个数据开始，每隔两个数据quyige
# 进行上述处理之前对数据进行截断处理，使数据长度为3的倍数+1，横轴数据和纵轴数据都进行截断处理
# bin=n的时候，前n//2个数据不要，后面每隔n-1个数据取一个数据，数据长度截断为n的倍数+n//2


def ch_bin(x, y, bin):
    N = len(x)
    relen = N // bin * bin
    re_x = x[:relen]
    re_y = y[:relen]
    res_x = []
    res_y = []
    i = 0
    while (True):
        if i <= bin // 2:
            i += 1
            continue
        else:
            res_x.append(re_x[i])
            i += bin
        if i >= relen - 1:
            break
    num = relen // bin
    for i in range(num):
        sum = 0
        for j in range(bin):
            sum += re_y[j + i * bin]
        res_y.append(sum * 1.0 / bin)
    if bin == 3:
        return (res_x, res_y[1:])
    else:
        return (res_x, res_y)
```