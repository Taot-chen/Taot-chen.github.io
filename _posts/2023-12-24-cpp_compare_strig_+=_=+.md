---
layout: post
title: cpp_compare_string_+=_=+
date: 2023-12-24
tags: [c_cpp]
author: taot
---

## std::string +=  && =+ compare

C++ 中，string 的拼接运算，常用的方法有 **先拼接再赋值`=+`** 和 直接使用**累加赋值运算符 `+=`**

在 Windows10 下，使用 mingw64 8.1.0 ，测试对比了 字符串使用着两种方法的效率差异。

实验结果显示，在该操作重复次数比较大时，**累加赋值运算符 `+=` 的效率明显高于先拼接再赋值`=+`**。顺便测试了 `int` 型这两种累加方式的效率，在测试的累计额范围内，两者没有差别。这里的测试累计额次数为 `200000`，每个测试结果取 10 的平均值。当累计额次数更多的时候，字符串的两种操作的结果效率差异会更加明显。

test code：

```cpp
#include<iostream>
#include<time.h>

int main(){
    std::string str1 = "a";
    std::string str2 = "a";
    double total_time = 0;
    int test_num = 10;

    // += for string
    for (int cnt = 0; cnt < test_num; cnt++) {
        clock_t start1 = clock();
        for (int i = 0; i < 200000; i++) {
            str1 += "a";
        }
        clock_t end1 = clock();
        total_time += (double)(end1 - start1) / CLOCKS_PER_SEC;
    }
    std::cout << "time of += for string: " << total_time / test_num << " s" << std::endl;

    // =+ for string
    total_time = 0;
    for (int cnt = 0; cnt < test_num; cnt++) {
        clock_t start2 = clock();
        for (int i = 0; i < 200000; i++) {
            str2 = str2 + "a";
        }
        clock_t end2 = clock();
        total_time += (double)(end2 - start2) / CLOCKS_PER_SEC;
    }
    std::cout << "time of =+ for string: " << total_time / test_num << " s" << std::endl;
    
    int a = 0;
    int b = 0;

    // += for int
    total_time = 0;
    for (int cnt = 0; cnt < test_num; cnt++) {
        clock_t start3 = clock();
        for (int i = 0; i < 200000; i++) {
            a += 1;
        }
        clock_t end3 = clock();
        total_time += (double)(end3 - start3) / CLOCKS_PER_SEC;
    }
    std::cout << "time of += for int: " << total_time / test_num << " s" << std::endl;
    
    // =+ for int
    total_time = 0;
    for (int cnt = 0; cnt < test_num; cnt++) {
        clock_t start4 = clock();
        for (int i = 0; i < 200000; i++) {
            b = b + 1;
        }
        clock_t end4 = clock();
        total_time += (double)(end4 - start4) / CLOCKS_PER_SEC;
    }
    std::cout << "time of =+ for int: " << total_time / test_num << " s" << std::endl;

    return 0;
}

```

result：

![string+=_test](https://image-hosting-taot.oss-cn-shanghai.aliyuncs.com/markdown-image202304050222659.png)

