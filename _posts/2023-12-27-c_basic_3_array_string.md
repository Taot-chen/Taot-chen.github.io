---
layout: post
title: c_basic_3_array_string
date: 2023-12-27
---

## 数组和字符串

### 1、概述

* 数组是在内存中连续的相同类型的变量空间
* 数组属于构造数据类型

```c
//数组定义
//数据类型 数组名[元素个数]={值1，值2，值3，……};
int arr[10]={9,8,7,6,5,4,3,2,1,0};

//元素访问
//数组名[下标]
//数组下标从0开始
for(int i=0;i)
{
    printf("%d\n",arr[i]);
}

//数组在内存中的存储方式和大小
for(int i=0;i<10;i++)
{
    printf("%p\n",&arr[i]);
}

printf("%p\n",&arr);

printf("数组在内存中占的大小：%d\n",sizeof(arr));
printf("数组元素在内存中占的大小：%d\n",sizeof(arr[0]));
printf("数组元素个数：%d\n",sizeof(arr)/sizeof(arr[0]));
```

* 数组在内存中是连续存储的，地址连续
* 数组名是一个指向数组首地址的地址常量
* 数组在内存中占的大小：数组类型*元素个数

### 2、数组的定义和使用

```c
# 数组定义的方式
int arr1[10]={9,8,7,6,5,4,3,2,1,0};
int arr2[]={9,8,7,6,5,4,3,2,1,0};		//自动得到数组长度为10
int arr3[10]={1,2,3};		// 前三个分别为1 2 3 ，其余为0
int arr4[10]={0};			//所有元素都为0
int arr5[10]={1};			//第一个为1，其余为0

int arr6[10];				//定义了没有赋值，内部的数据不确定，一般会乱码
arr6[0]=1;					//第一个为1，其余依然乱码

int arr[];					//在这种写法不可以

//这种写法不可以，数组定义的时候，数组长度必须是常量，或者常量表达式
//数组必须预先知道大小
//动态数组	开辟堆空间
int i=10;
int arr[i];


//从键盘获取数组元素值
int arr7[10];
for(int i=0;i<10;i++){
    scanf("%d",&arr7[i]);
}


```

* 数组下标越界，可能会报错，但是越界后，访问到的内容不确定。下标越界，编译器不会报错

### 3、数组逆置

```c
int arr[10]={1,2,3,4,5,6,7,8,9,10};
int i=0;
int j=sizeof(arr)/sizeof(arr[0]);
while(i<j)
{
    //通过临时变量交换数据
    int temp=arr[i];
    arr[i]=arr[j];
    arr[j]=temp;
    i++;
    j--;
}

```

### 4、冒泡排序

```c
//内层比较次数为：元素个数-1-执行次数
//外层执行次数为：元素个数-1
int arr[]={9,1,5,7,2,10,8,4,6,3};
for(int i=0;i<10-1;i++)
{
	for(int j=0;i<10-1-i;j++)
    {
        if(arr[j]>arr[j+1])
        {
            //通过大于或者小于号控制升序或者降序
            int temp=arr[j];
            arr[j]=arr[j+1];
            arr[j+1]=temp;
        }
    }
}
```

### 5、二维数组

#### 5.1、二维数组的定义和使用

```c
/*
// 一维数组：数据类型 数组名[元素个数]={值1，值2，……}
// 二维数组：数据类型 数组名[行数][列数]={{值1，值2，……},{值1，值2，……},{值1，值2，……},……}
*/
int arr[2][3]={{1,2,3},{4,5,6}};

arr[1][2]=20;
for(int i=0;i<2;i++)
{
	for(int j=0;j<3;j++)
    {
        printf("%d ",arr[i][j]);
    }
    printf("\n");
}

printf("二维数组的大小：%d\n",sizeof(arr));
printf("二维数组一行的大小：%d\n",sizeof(arr[0]));
printf("二维数组的元素大小：%d\n",sizeof(arr[0][0]));

printf("二维数组的行数：%d\n",sizeof(arr)/sizeof(arr[0]));
printf("二维数组的列数：%d\n",sizeof(arr[0])/sizeof(arr[0][0]));

//二维数组首地址
//下面三个语句的输出结果一样
printf("%p\n",arr);
printf("%p\n",arr[0]);
printf("%p\n",&arr[0][0]);

//数组的第二行的首地址
printf("%p\n",arr[1]);

//二维数组初始化
int arr[2][2]={1,2,3,4,5,6};	//等价于 int arr[2][3]={{1,2,3},{4,5,6}}; 但是一般不要这么写，不方便维护

//int arr[2][3]
int arr[][3]={1,2,3,4,5,6};	

//int arr[3][3]
int arr[][3]={1,2,3,4,5,6,7};	

//二位数组初始化的时候，列数不能省略，只能省略行


```

#### 5.2、二维数组的应用

```c
//定义一个二位数组，存储五名学生三门成绩，arr[5][3]
//求出每名学生的总成绩，平均成绩
//求出每门课的总成绩，平均成绩
int arr[5][3];

//输入成绩
for(int i=0;i<5;i++)
{
    for(int j=0;j<3;j++)
    {
        scanf("%d",&arr[i][j]);
    }
}
//输出成绩
for(int i=0;i<5;i++)
{
    for(int j=0;j<3;j++)
    {
        printf("%d",&arr[i][j]);
    }
}

//计算每名学生的总成绩和平均成绩
for(int i=0;i<5;i++)
{
    int sum=0;
    for(int j=0;j<3;j++)
    {
        sum+=arr[i][j];
    }
    int average=sum/3;
}

//计算每门课的总成绩和平均成绩
for(int i=0;i<3;i++)
{
    int sum=0;
    for(int j=0;j<5;j++)
    {
        sum+=arr[i][j];
    }
    int average=sum/5;
}
```

### 6、多维数组

#### 6.1、多维数组的定义

```c
// 一维数组：数据类型 数组名[元素个数]={值1，值2，……}
// 二维数组：数据类型 数组名[行数][列数]={{值1，值2，……},{值1，值2，……},{值1，值2，……},……}
//多维数组声明：数组类型 数组名[n1][n2]……[nn];
int arr[2][3][4]={{{1,2,3,4},{2,3,4,5},{3,4,5,6}},{{4,5,6,7},{5,6,7,8},{6,7,8,9}}};
for(int i=0;i<2;i++)
{
	for(int j=0;j<3;j++)
    {
        for(int k=0;k<4;k++)
        {
            printf("%d ",arr[i][j][k]);
        }
        printf("\n");
    }
    printf("\n");
}


//三维数组的大小
printf("三维数组的大小：%d\n",sizeof(arr));
printf("三维数组一层的大小：%d\n",sizeof(arr[0]));
printf("三维数组一行的大小：%d\n",sizeof(arr[0][0]));
printf("三维数组元素的大小：%d\n",sizeof(arr[0][0][0]));

//三维数组的层数
printf("三维数组的层数：%d\n",sizeof(arr)/sizeof(arr[0]));
printf("三维数组的行数：%d\n",sizeof(arr[0])/sizeof(arr[0][0]));
printf("三维数组的列数：%d\n",sizeof(arr[0][0])/sizeof(arr[0][0][0]));

//三维数组定义的时候，层数可以省略
//高维数组定义的候，第一个维度可以省略
int arr[][3][4]={0}

//四维数组
int arr[2][3][4][5]={1,2,3};
//元素个数：2*3*4*5
```

### 7、字符数组和字符串

#### 7.1、字符数组与字符串的区别

```c
//定义字符数组
char arr[6]={'h','e','l','l','o','0'};	//此时不是字符串，只是字符数组
char arr2[5]={'h','e','l','l','o'};	
//字符
char ch='h';
//字符串	字符串结束标志位 \0
char *str="hello";
char arr1[]={'h','e','l','l','o','\0'};		//此时就是字符串了，数字0等同于 \0,，但是不等同于 '0'
printf("%s",arr1);
printf("%s",arr2);		//不会报错，但是输出会乱码

printf("%d\n",sizeof(arr));

//可以通过字符的ASCII码进行字符数组的初始化，但是下面的初始化末尾没有'\0'，因此输出完前7个字符之后，后面会乱码，直到遇到内存中的'\0'为止
char ch3[]={110,111,112,101,123,98,99};
printf("%s",ch3);

char ch4[]={"hello"};	//也可以，不要大括号也可以

```

#### 7.2、字符数组存储字符串

```c
//定义字符数组，存储字符串
char ch[10];

//输入：helloworld ，会报错
//输入：helloworl ，会正常输入输出
//输入：hellowo ，会正常输入输出
//输入：hello wor，会输出hello
scanf("%s",ch);		//在输入字符串的时候，长度超过9个会报错

printf("%s",ch);
```

* 拼接两个字符串

```c
char ch1[]="hello";
char ch2="world";
char ch3[30];

//字符串拼接
int i=0;
int j=0;
while(ch1[i]!='\0')
{
    ch3[i]=ch1[i];
    i++;
}
while(ch2[j]!='\0')
{
    ch3[i+j]=ch2[j];
    j++;
}
ch3[i+j]='\0';

printf("%s",ch3);
```

#### 7.3、字符串输入输出

```c
char ch[10];

//可以在输入的时候对字符串长度进行限定，这里输入超过9个字符就不会报错，但是实际输入只有九个字符，输出也只有9个字符
scanf("%9s",ch);		
printf("%s",ch);

```

1）gets() 函数：从标准输入读入字符，并保存到指定的内存空间，直到出现换行符或读到文件结尾为止

```c
char *gets(char *s);

```

* 参数：s，字符串首地址
* 返回值：成功，读入的字符串；失败，NULL
* gets(str) 和 scanf("%s",str) 的区别：
    * gets(str) 允许输入的字符串含有空格
    * scanf("%s",str) 不允许含有空格

```c
# include <stdio.h>
char ch[100];
char ch2[100];

//通过键盘获得一个字符串
//gets() 接收字符串可以带空格
//scanf 可以通过正则表达式来接收空格
scanf("%[^\n]",ch);
gets(ch);
printf("%s",ch);
```

**注：由于 scanf() 和 gets() 无法知道字符串的大小，必须遇到换行符或者读到文件结尾为止才接收输入，因此容易导致数组越界（缓冲区溢出）的情况。**

2）fgets() 函数：从 stream 指定的文件内读入字符，保存到所指定的内存空间，直到出现换行字符、读到文件末尾或是读了 size-1 个字符为止，最后会自动加上 '\0' 字符作为结束标志。

```c
# include<stdio.h>
char *fgets(char *s, int size,FILE *stream);
```

* 参数：
    * s，字符串；
    * size，指定最大读取字符串的长度（size-1）
    * stream：文件指针，如果读取键盘输入的字符串，固定写作 stdin
* 返回值：
    * 成功：成功读取的字符串
    * 读到文件尾或者出错：NULL
* fgets() 在读取一个用户通过键盘输入的字符串的时候，同事把用户输入的回车也作为字符串的一部分。通过 gets() 和 scanf() 输入一个字符串的时候，不包含结尾的 '\n'，**但通过 fgets() 结尾多了 '\n'**。
* fgets() 函数是安全的，不存在缓冲区溢出的问题
* fgets() 函数可以接收空格和换行
* fgets() 函数获取字符串长度短于元素个数会有 '\n'，大于则没有

```c
char ch[10];

fgets(ch,sieof(ch),stdin);
printf("%s",ch);
```

3）puts () 函数：从标准输出设备输出字符串，在输出完成后自动输出一个 '\n'

```c
#include<srdio.h>
int puts(const char *s);
```

* 参数：s，字符串首地址
* 返回值：成功，非负数；失败，-1

```c
char ch[]="hello world";

//puts() 自带换行
puts(ch);

puts("hello\0 world");		//输出结果为hello，输出的时候遇到 '\0' 停止

```

4）fputs() 函数：将 str 所指定的字符串写入到 stream 指定的文件中，字符串结束符 '\0' 不写入文件

```c
#include<stdio.h>
int fputs(const char *str,FILE *stream);
```

* 参数：
    * str，字符串
    * stream，文件指针，如果把字符串输出到屏幕，固定写为 stdout
* 返回值：成功，0；失败，-1
* fputs() 是 puts() 的文件操作版本，但是 fputs() 不会自动在末尾输出一个 '\n'

```c
char ch[]="hello world";
fputs(ch,stdout);
```

#### 7.4、字符串的长度

1）strlen() 函数：计算指定字符串 s 的长度，不包含字符串结束符 '\0'

```c
#include <string.h>
size_t strlen(const char *s);
```

* 参数：s，字符串首地址
* 返回值：字符串 s 的长度，size_t 为 unsigned int 类型

```c
//计算字符串有效长度
char ch[100]="hello world";
printf("数组大小：%d",sizeof(ch));		//100
printf("字符串大小：%d",strlen(ch));		//11

char ch1[]="hello world";
printf("数组大小：%d",sizeof(ch1));		//12
printf("字符串大小：%d",strlen(ch1));		//11
```

```c
char ch1[]="hello world";
int len=0;

while(ch[len]!='\0')
{
	len++;
}
printf("字符串的长度：%d"，len);
```
