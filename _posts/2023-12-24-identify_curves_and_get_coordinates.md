---
layout: post
title: identify_curves_and_get_coordinates
date: 2023-12-24
---

## 识别图片中曲线并获取其坐标

**思路：**
</br>
&emsp;1）通过图像算法中常用的边界识别的方法来识别曲线；</br>
&emsp;2）根据曲线上每一点的像素坐标和坐标轴的数值范围，来计算曲线上每一个像素点在坐标轴中的像素坐标。

**实现过程：**

### 一、曲线识别

#### 1）图片预处理

**思路：**</br>
&emsp;将待处理的图像转换成灰度图，在转换成二值图像；対二值图像的每一行和每一列的像素求和，根据像素和识别出图像的坐标轴范围；在坐标轴范围内遍历二值图像的每一列，将每列中像素值为0的像素下面的像素值全部置零。

**原图：**</br>
![原图]("图片路径", "原图")

**具体实现：**
</br>
将图像转换成二值图像：

```
    # 打开图片
    img = cv.imread(pic_name)
    # 灰度化
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # 二值化
    # 此时的第二个和第三个参数，
    # 即二值化的上下阈值需要根据图片的实际情况个来调整
    ret, binary_img = cv.threshold(gray_img, 180, 255, cv.THRESH_BINARY)
```

识别坐标轴的范围：

```
    # 图像像素按行和列求和
    # 二值图像反相
    con_bi_img = 255 - binary_img
    column_sum_img = np.sum(con_bi_img, axis=0)
    row_sum_img = np.sum(con_bi_img, axis=1)
    
    # 排序
    sort_column_sum = np.sort(column_sum_img)
    sort_column_sum_indices = np.argsort(column_sum_img)
    sort_row_sum = np.sort(row_sum_img)
    sort_row_sum_indices = np.argsort(row_sum_img)
    
    print("列：\n")
    print(sort_column_sum[len(sort_column_sum) - 5:])
    print(sort_column_sum_indices[len(sort_column_sum_indices) - 5:])
    print("行：\n")
    print(sort_row_sum[len(sort_row_sum) - 5:])
    print(sort_row_sum_indices[len(sort_row_sum_indices) - 5:])
```

将曲线和坐标轴之间的区域设置成黑色：

```
for i in range(45, 773):
        flag = 0
        for j in range(73, 495):
            if binary_img[j][i] == 0:
                flag += j
                break
        for j in range(flag, 495):
            binary_img[j][i] = 0
```

**预处理结果：**</br>
![预处理结果]("图片路径", "预处理结果")

#### 2）识别曲线

**思路：**</br>
&emsp;通过对预处理的到图像进行边缘获取；识别出图像的坐标轴，在坐标轴范围内的边缘线就是所要识别的曲线。

**具体实现：**
</br>
边缘提取：

```
    # 边缘提取
    xgrd = cv.Sobel(binary_img, cv.CV_16SC1, 1, 0)
    ygrd = cv.Sobel(binary_img, cv.CV_16SC1, 0, 1
    egde_output = cv.Canny(xgrd, ygrd, 50, 150)
```

曲线获取：

```
    # 图像像素按行和列求和
    column_sum_img = np.sum(egde_output, axis=0)
    row_sum_img = np.sum(egde_output, axis=1)
    
    # 排序
    sort_column_sum = np.sort(column_sum_img)
    sort_column_sum_indices = np.argsort(column_sum_img)
    sort_row_sum = np.sort(row_sum_img)
    sort_row_sum_indices = np.argsort(row_sum_img)
    
    print(sort_column_sum[len(sort_column_sum) - 10:])
    print(sort_column_sum_indices[len(sort_column_sum_indices) - 10:])
    print(sort_row_sum[len(sort_row_sum) - 10:])
    print(sort_row_sum_indices[len(sort_row_sum_indices) - 10:])

    fc = egde_output[71:494, 47:770]
```

**曲线识别结果：**</br>
![曲线识别结果]("图片路径", "曲线识别结果")

### 二、坐标计算

**思路：**
</br>
&emsp;根据原图的坐标范围与像素的对应关系来计算曲线上每一个像素的数值坐标。

**具体实现：**
</br>

```
    # 提取图像数据
    rows = (fc.shape)[0]
    cols = (fc.shape)[1]

    min_x = 4000
    max_x = 400
    min_y = 0.0
    max_y = 0.95

    x_axis = np.empty([rows, cols])
    y_axis = np.empty([cols, rows])

    # x_interval和y_interval用于调整数据长度，在报错的时候可以通过他们来调整
    x_interval = (max_x - min_x) / (cols + 1)
    x_value = np.arange(min_x + x_interval, max_x, x_interval)
    y_interval = (max_y - min_y) / (rows + 1)
    y_value = np.arange(max_y - y_interval, min_y, -y_interval)

    x_axis[:, ] = x_value
    y_axis[:, ] = y_value
    y_axis = y_axis.T

    x_fc = x_axis.T[fc.T == 255]
    y_fc = y_axis.T[fc.T == 255]
```

x_fc 和 y_fc 即为所求曲线的横坐标和纵坐标
</br>

**曲线获取结果：**</br>
![曲线获取结果]("图片路径", "曲线获取结果")

**完整代码：**</br>
&emsp;为方便代码修改和阅读，上文中的几个步骤的实现分别在独立的文件中，共有三个文件。

**./main.py**</br>

```
import get_curve as gc

pic_name = "the path of the original picture"
res = gc.get_points(pic_name)
```

**./pic_prepro.py**</br>

```
import numpy as np
import cv2 as cv


def pre_pro(pic_name):
    # 打开图片
    img = cv.imread(pic_name)
    # 灰度化
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # 二值化
    # 此时的第二个和第三个参数，
    # 即二值化的上下阈值需要根据图片的实际情况个来调整
    ret, binary_img = cv.threshold(gray_img, 180, 255, cv.THRESH_BINARY)
    cv.imshow('binary_img', binary_img)
    cv.waitKey(0)
    cv.destroyAllWindows()

    # 图像像素按行和列求和
    # 二值图像反相
    con_bi_img = 255 - binary_img
    # print(np.shape(binary_img))
    cv.imshow('con_bi_img', con_bi_img)
    cv.waitKey(0)
    cv.destroyAllWindows()
    column_sum_img = np.sum(con_bi_img, axis=0)
    row_sum_img = np.sum(con_bi_img, axis=1)
    # 排序
    sort_column_sum = np.sort(column_sum_img)
    sort_column_sum_indices = np.argsort(column_sum_img)
    sort_row_sum = np.sort(row_sum_img)
    sort_row_sum_indices = np.argsort(row_sum_img)
    print("列：\n")
    print(sort_column_sum[len(sort_column_sum) - 5:])
    print(sort_column_sum_indices[len(sort_column_sum_indices) - 5:])
    print("行：\n")
    print(sort_row_sum[len(sort_row_sum) - 5:])
    print(sort_row_sum_indices[len(sort_row_sum_indices) - 5:])

    # 通过每一列和每一行像素和的输出值，判断坐标轴的位置
    # 在坐标轴范围内遍历每一列，将曲线下面的部分设置成黑色
    for i in range(45, 773):
        flag = 0
        for j in range(73, 495):
            if binary_img[j][i] == 0:
                flag += j
                break
        for j in range(flag, 495):
            binary_img[j][i] = 0
    cv.imshow('binary_img', binary_img)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return binary_img
```


**./get_curve.py**

```
import numpy as np
from numpy.core.fromnumeric import shape
import cv2 as cv
import pic_prepro as pp


def get_points(picture_name):

    binary_img = pp.pre_pro(picture_name)
    cv.imshow('binary_img', binary_img)
    cv.waitKey(0)
    cv.destroyAllWindows()

    # 边缘提取
    xgrd = cv.Sobel(binary_img, cv.CV_16SC1, 1, 0)
    ygrd = cv.Sobel(binary_img, cv.CV_16SC1, 0, 1)

    egde_output = cv.Canny(xgrd, ygrd, 50, 150)
    cv.imshow('canny_edge', egde_output)
    cv.waitKey(0)
    cv.destroyAllWindows()

    # 图像像素按行和列求和
    column_sum_img = np.sum(egde_output, axis=0)
    row_sum_img = np.sum(egde_output, axis=1)
    # 排序
    sort_column_sum = np.sort(column_sum_img)
    sort_column_sum_indices = np.argsort(column_sum_img)
    sort_row_sum = np.sort(row_sum_img)
    sort_row_sum_indices = np.argsort(row_sum_img)
    print(sort_column_sum[len(sort_column_sum) - 10:])
    print(sort_column_sum_indices[len(sort_column_sum_indices) - 10:])
    print(sort_row_sum[len(sort_row_sum) - 10:])
    print(sort_row_sum_indices[len(sort_row_sum_indices) - 10:])

    fc = egde_output[71:494, 47:770]
    cv.imshow('function_curve', fc)
    cv.waitKey(0)
    cv.destroyAllWindows()

    # 提取图像数据
    rows = (fc.shape)[0]
    cols = (fc.shape)[1]

    min_x = 4000
    max_x = 400
    min_y = 0.0
    max_y = 0.95

    x_axis = np.empty([rows, cols])
    y_axis = np.empty([cols, rows])

    # x_interval和y_interval用于调整数据长度，在报错的时候可以通过他们来调整
    x_interval = (max_x - min_x) / (cols + 1)
    x_value = np.arange(min_x + x_interval, max_x, x_interval)
    y_interval = (max_y - min_y) / (rows + 1)
    y_value = np.arange(max_y - y_interval, min_y, -y_interval)

    x_axis[:, ] = x_value
    y_axis[:, ] = y_value
    y_axis = y_axis.T

    x_fc = x_axis.T[fc.T == 255]
    y_fc = y_axis.T[fc.T == 255]

    return (x_fc, y_fc)
```