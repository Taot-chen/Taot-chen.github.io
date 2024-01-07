---
layout: post
title: identify_circles_and_get_average_radius
date: 2023-12-24
tags: [algorithm]
author: taot
---

## 识别图片中的圆形并求所有圆形的平均半径
**思路：**</br>
&emsp;1）对图片进行预处理；</br>
&emsp;2）利用opencv，借助霍夫梯度法识别图像中的圆形；</br>
&emsp;3）在原图像中标记出识别到的圆形，并计算识别到的圆的平均半径。</br>

**实现过程：**</br>
**待识别图像：**</br>
![待识别图像]("图片路径", "待识别图像")

### 一、图像预处理

**思路：**</br>
&emsp;对图像进行epf边缘滤波，减小圆形识别的干扰；将图像转换成灰度图像。</br>

**具体实现：**</br>
epf边缘滤波：

```
temp = cv.pyrMeanShiftFiltering(img, 2, 30)
```

灰度处理：

```
cimg = cv.cvtColor(temp, cv.COLOR_RGB2GRAY)
```

### 二、圆形识别

**思路：**</br>
&emsp;利用opencv的霍夫梯度法识别图像中的圆形。</br>

**具体实现：**</br>

```
circles = cv.HoughCircles(cimg,
                              cv.HOUGH_GRADIENT,
                              1,
                              20,
                              param1=50,
                              param2=30,
                              minRadius=20,
                              maxRadius=50)
circles = np.uint16(np.around(circles))
for i in circles[0, :]:
    cv.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)  # 圆
    cv.circle(img, (i[0], i[1]), 2, (255, 255, 0), 2)  # 圆心
```

**识别结果：**</br>
![识别结果]("图片路径", "识别结果")

### 三、平均半径计算

**思路：**</br>
&emsp;根据比例尺计算出每个像素对应的、长度，再依次得到每个识别到的圆的半径，最后得到平均半径。</br>

**具体实现：**</br>
比例尺的计算：

```
    # ruler计算
    ruler_img = cv.imread(图像中比例尺的单独截图路径)
    ruler_size = np.shape(ruler_img)
    length = 0
    for i in range(len(ruler_size)):
        if length < ruler_size[i]:
            length = ruler_size[i]
    l_cons = 78
    delta_pixle = l_cons / length
```

平均半径的计算：

```
    # 平均半径计算
    cnt = 0
    sum_r = 0
    x = []
    y = []
    radius = []
    for circle in circles:
        cnt += 1
        sum_r += circle[2]
        x.append(circle[0])
        y.append(circle[1])
        radius.append(circle[2])
    ava_radius = sum_r / cnt
    ava_radius *= delta_pixle
    # 圆心坐标
    for i in range(len(x)):
        x[i] *= delta_pixle
    for i in range(len(y)):
        y[i] *= delta_pixle
```

**完整代码：**</br>
&emsp;为方便代码的修改，把各个单独的功能的代码放在了独立的文件中，文件结构如下图：</br>
**文件结构：**</br>
![文件结构]("图片路径", "文件结构")

**./main.py**

```
import sys

sys.path.append(
    "/home/taot/Python/Useful_tools/circle_analyse_1/assistant_func")
import read_preprocess as rp
import circle_detection as cd
import cal_ava_radius as car
import cv2 as cv

pic = rp.pre_pro("/home/taot/Python/Useful_tools/circle_analyse_1/data.jpg")
img = pic[0]
cimg = pic[1]

ans = cd.cir_dec(img, cimg)
res = ans[0]
cv.imwrite("/home/taot/Python/Useful_tools/circle_analyse_1/res.png", res)

circles = ans[1][0]
ava_radius = car.ava_r(circles)
print("The avarage radius is: {}微米".format(ava_radius))
```

**./read_preprocess.py**

```
import cv2 as cv
import numpy as np


def pre_pro(pic_name):
    img = cv.imread(pic_name)
    cv.imshow("input img", img)
    cv.waitKey(0)
    cv.destroyAllWindows()

    # epf边缘滤波
    temp = cv.pyrMeanShiftFiltering(img, 2, 30)
    cv.imshow("epf", temp)
    cv.waitKey(0)
    cv.destroyAllWindows()
    # 灰度处理
    cimg = cv.cvtColor(temp, cv.COLOR_RGB2GRAY)
    cv.imshow("gray", cimg)
    cv.waitKey(0)
    cv.destroyAllWindows()

    return (img, cimg)
```


**./circle_detection.py**

```
import sys

sys.path.append(
    "/home/taot/Python/Useful_tools/circle_analyse_1/assistant_func")
import read_preprocess as rp
import cv2 as cv
import numpy as np


def cir_dec(img, cimg):
    circles = cv.HoughCircles(cimg,
                              cv.HOUGH_GRADIENT,
                              1,
                              20,
                              param1=50,
                              param2=30,
                              minRadius=20,
                              maxRadius=50)
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        cv.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)  # 圆
        cv.circle(img, (i[0], i[1]), 2, (255, 255, 0), 2)  # 圆心
    cv.imshow("res", img)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return (img, circles)
```


**./cal_ava_radius.py**

```
import numpy as np
import cv2 as cv


def ava_r(circles):
    # ruler计算
    ruler_img = cv.imread(
        "/home/taot/Python/Useful_tools/circle_analyse_1/ruler.png")
    ruler_size = np.shape(ruler_img)
    length = 0
    for i in range(len(ruler_size)):
        if length < ruler_size[i]:
            length = ruler_size[i]
    l_cons = 78
    delta_pixle = l_cons / length

    # 平均半径计算
    cnt = 0
    sum_r = 0
    x = []
    y = []
    radius = []
    for circle in circles:
        cnt += 1
        sum_r += circle[2]
        x.append(circle[0])
        y.append(circle[1])
        radius.append(circle[2])
    ava_radius = sum_r / cnt
    ava_radius *= delta_pixle
    for i in range(len(x)):
        x[i] *= delta_pixle
    for i in range(len(y)):
        y[i] *= delta_pixle
    np.savetxt("/home/taot/Python/Useful_tools/circle_analyse_1/x.txt", x)
    np.savetxt("/home/taot/Python/Useful_tools/circle_analyse_1/y.txt", y)
    np.savetxt("/home/taot/Python/Useful_tools/circle_analyse_1/radius.txt",
               radius)
    return ava_radius
```