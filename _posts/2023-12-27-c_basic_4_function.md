---
layout: post
title: c_basic_4_function
date: 2023-12-27
tags: [c_cpp]
author: taot
---

## 函数

### 1、概述

#### 1.1、函数分类

1）系统函数（库函数）

2）用户定义函数

#### 1.2、函数的作用

* 降低代码重复率
* 让程序更加模块化，利于阅读、修改和完善

#### 1.3、函数的调用：随机数

函数调用 5 要素：

* 头文件：包含指定的头文件
* 函数名字：函数名字必须与头文件声明的名字一样
* 功能：要知道函数实现的功能
* 参数：参数类型要匹配
* 返回值：根据需要接收返回值

```c
#include<time.h>
time_t time(time_t *t);
//功能：获取系统当前时间
//参数：常设置为NULL
//返回值：当前系统时间，time_t 相当于 long 类型，单位为毫秒

# include<stdlib.h>
void srand(unsigned int seed);
//功能：用来产生 rand() 函数产生随机数时的随机种子
//参数：如果每次 seed 相同，rand() 产生的随机数相同
//返回值：无

# include<stdlib.h>
int rand(void);
//功能：返回一个随机数
//参数：无
//返回值：随机数

```

```c
#include<time.h>
# include<stdlib.h>
# include<stddio.h>
//双色球	6个红球，1-32，不可重复		1个蓝球，1-16，可以和红球号码重复
int arr[7]={0};
srand((unsigned int)time(NULL));
int value=0;
int flag=0;
int j;
for(int i=0;i<6;i++)
{
	value=rand()%32+1;
    //去重
    for(j=0;j<flag;j++)
    {
        if(value==arr[j])
        {
            i--;
            break;
        } 
    }
    if(j==flag)
    {
        arr[flag]=value;
    	flag++;
    }
    
}
arr[7]=rand()%16+1;
```

### 2、函数的定义

#### 2.1、函数的定义格式

```c
返回类型 函数名(形式参数列表)
{
	函数体（数据定义部分，执行语句部分）
}
```

```c
//函数定义
int add(int a,int b)
{
    int sum=a+b;
    return sum;
}

int main()
{
    int a=10;
    int b=20;
    
    //函数调用
    int res=add(a,b);
    return 0;
}
```

* 在不同函数中的可以有相同的变量名，因为作用域不同；
* 在函数调用过程中，传递的参数称为实参（实际参数），有具体的值；
* 在函数定义过程中的参数称为形参（形式参数），只有类型，没有具体的值；
* 在函数调用过程中，将实参传递给形参；
* 在函数调用结束，函数相关的内存会在栈区自动销毁。

#### 2.2、函数名、形参、实参、返回值

1）函数名：见名知意

2）形参列表：在函数没有被调用时，形参不占用内存中的存储单元

* 在定义函数时指定的形参，可有可无，根据函数的需要来设计，如果没有形参，括号内为空，或者写一个 void 关键字

3）函数体：函数功能实现的过程

4）返回值：通过函数中的 return 语句获得，return 后面的值也可以是一个表达式

* 尽量保证 return 中的数据类型和函数的返回类型一致
* 如果函数返回的数据类型和函数返回类型不一致，则以函数返回类型为准，对于数值类型，会自动进行数据类型转换

### 3、函数的调用

#### 3.1、函数的执行流程

1）进入 main() 函数

2）调用自定义函数

3）自定义函数执行完，main() 会继续往下执行，直到执行 return 0，程序执行完毕

#### 3.2、函数的形参和实参

* 形参出现在函数定义中，在整个函数体中都可以使用，离开函数则不能使用；
* 实参出现在主调函数中，进入被调函数后，实参也不能使用；
* 实参变量对形参变量的数据传递是单向的值传递，只能由实参传递给形参，而不能由形参传递给实参；
* 再调用函数时，编译系统临时给形参分配存储单元。调用结束后，形参单元被释放；
* 形参与实参在内存中存放在不同的存储单元。在执行一个调用过程中，形参的值发生改变，不会影响实参的值。

```c
//字符串比较，相同返回0，不同返回 1 或者 -1
int strcmp(char ch1[],char ch2[])
{
    int i=0;
    while(ch1[i]==ch2[i])
    {
        //是否到字符串结尾
        if(ch1[i]=='\0')
        {
            return 0;
        }
        i++;
    }
    return ch1[i]>ch2[i]?1:-1;
}
```

### 4、函数样式

#### 4.1、无参函数

* 无参函数调用，直接函数名后跟括号即可

```c
//无参函数
void func1()
{
	printf("Hello\n");
}

int func2()
{
    return rand()%10;
}

int main()
{
    func1;
    srand((unsigned int)time(NULL));
    int randNum=func2();
    return 0;
}
```

#### 4.2、有参函数

1）如果实参列表包含多个实参，则各参数间用逗号隔开；

2）实参与形参应该个数相等，类型匹配（相同或者赋值类型兼容），按顺序对应；

3）实参可以是常量、变量、表达式，无论实参是何种类型的量，在进行函数的调用时，他们应该具有确定的值。

```c
//有参函数
//冒泡排序函数
void bubbleSort(int arr[],int len)
{
	for(int i=0;i<len-1;i++)
    {
        for(int j=0lj<len-1-i;j++)
        {
            if(arr[j]>arr[j+1])
            {
                int temp=arr[j];
                arr[j]=arr[j+1];
                arr[j+1]=temp;
            }
        }
    }
    return;		//无返回值，return可写可不写
}

int main()
{
	int arr[]={9,1,7,4,2,10,3,8,6,5};
    bubbleSort(arr,sizeof(arr)/sizeof(arr[0]));
    return 0;
    
}
```

### 5、函数的返回值

* 如果函数定义没有返回值，函数调用时不能写 void 关键字，调用函数时也不能接收函数的返回值；
* void，空类型：void 类型不可以直接定义数据，但可以作为函数的返回值类型，表示没有返回值；

### 6、函数的声明

自定义函数三个过程：

* 函数声明；
* 函数定义；
* 函数调用

**当函数定义在主函数之前，可以直接定义，不比声明；但是当函数定义在主函数之后，就必须在主函数之前声明函数。**

```c
//函数声明
extern int add(int a,int b);
//在同一个文件中进行声明和定义，函数声明的时候可以省略 extern
int add(int a,int b);
//还可以省略形参的参数名
int add(int, int );

int main()
{
    //函数调用
    add(10,20);
	return 0;
}

//函数定义
int add(int a,int b)
{
    return a+b;
}
```

* 在一个程序中，函数只能定义一次，但是可以声明多次，调用多次；
* 函数声明时，可以省略形参的名称，只需要说明形参的类型即可。

### 7、主函数和 exit 函数

1）exit() 函数

* 功能：终止程序执行；
* 参数：整型，就是函数中止执行之后返回的错误码

```c
exit(0);
```

* 在主函数中，exit(0) 和 return 0 效果一样；但是在子函数中，调用 return 只是代表子函数中终止了，在子函数中调用 exit ，那么程序就终止了。

### 8、多文件编程

#### 8.1、分文件编程（多文件编程）

* 把函数声明放在头文件 xxx.h 中，在主函数中包含相应的头文件
* 在头文件对应的 xxx.c 中实现 xxx.h 声明的函数

01main.c

```c
#include<stdio.h>
#include"02func.h"

int main()
{
	int a = 10;
	int b = 20;
	int res = maxValue(a, b);
	printf("%d\n", res);
	return 0;
}
```

02func.c

```c
#include"02func.h"
//函数定义
int maxValue(int a, int b)
{
	return a > b ? a : b;
}
```

02func.h

```c
#pragma once	// 防止头文件重复包含
//#ifndef __02_FUNC_H__
//#define __02_FUNC_H__

//全局变量的定义		
//函数的声明
//extern int maxValue(int a, int b);
int maxValue(int a, int b);
//int maxValue(int,int);

```

#### 8.2、防止头文件重复包含

* 为避免同一个头文件被包含多次，C/C++中有两种方式：
    * #ifndef 方式；
    * #pragma 方式
