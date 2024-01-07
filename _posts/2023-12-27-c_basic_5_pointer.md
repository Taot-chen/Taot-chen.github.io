---
layout: post
title: c_basic_5_pointer
date: 2023-12-27
tags: [c_cpp]
author: taot
---

## 指针

### 1、概述

#### 1.1、内存

* 存储器：用来存储程序，辅助CPU进行运算处理的重要部分
* 内存：内部存储器，暂存程序、数据，掉电丢失，SRAM，DRAM，DDR，DDR2，DDR3……
* 外存：外部存储器，长时间保存程序、数据，掉电不丢，ROM，ERROM，FLASH，硬盘，光盘……

内存是沟通CPU与硬盘的桥梁：

* 暂存方CPU的运算数据；
* 暂存与硬盘等外部存储器交换的数据。

#### 1.2、物理存储器和存储地址空间

存储地址空间：对存储器编码的范围

* 编码：对每个物理存储单元（一个字节）分配一个编号；
* 寻址：可以根据分配的号码找到相应的存储单元，完成数据的读写。

#### 1.2、内存地址

* 将内存抽象成一个很大的一维字符数组；
* 编码就是对内存的每一个字节分配一个32位或64位的编号；
* 这个内存编号我们称为内存地址；
* char：占一个字节，分配一个地址；
* int：占四个字节分配四个地址，通常使用首地址；
* float，struct，函数，数组等。

```c
//int a=10;
int a =0xaabbccdd;

printf("%p\n",&a);
getchar();
```

* 在 Windows 中，进行数据存储的时候，采用小端对齐

```c
int a=10;
//定义指针变量存储地址
int* p;
p=&a;

printf("%p\n",&a);
printf("%p\n",p);

// 通过指针 p 也可以间接改变变量 a 的值
*p=100;



//指针在内存中大小
//指针的大小：4字节（32位），8字节（64位）
printf("%d\n",sizeof(int*));

//指针类型应该和数据类型对应
char ch='a';
char* p1=&ch;

//指针大小：4字节（32位），8字节（64位）
printf("%d\n",sizeof(p1));

```

* 通过变量 a 可以改变值，通过指针 p 也可以间接改变变量的值；
* 指针类型：数据类型*
* 指针类型应该和数据类型对应
* &是取地址符号，是升维度的；*是取值符号，是降维度的

#### 1.3、指针大小

* 所有的指针类型存储的都是内存地址
* 指针大小：4字节（32位），8字节（64位）

```c
char ch='a';
int* p=&ch;
//上面的写法没问题，&ch和p存储的都是 ch 的地址
//但是此时 *p 无法取到 ch 的值
//也无法通过*p来修改值
*p=123456;

//在定义指针的时候，滋指针类型一定要和变量类型对应上
//例如，通过智者间接地改变变量的值，int* 指针是改变四个字节内存储的内容，而 char* 指针改变的是一个字节内存储的内容，会出错
```

#### 1.4、野指针和空指针

```c
//野指针：指针变量指向一个未知的空间（未定义的空间）
int *p=10;		//内存地址编号为100的内存地址赋值给 p
printf("%d\n",*p);		//这里会报错，可能是这个内存无法被访问
```

* 野指针：指针变量指向一个未知的空间（未定义的空间）
* 任意数值赋值给指针变量是没有意义的，这样的指针会成为野指针，因为这样的指针指向的内存是未知的
* 操作野指针对应的内存空间可能报错
* 操作系统将编号为 0-255 的内存作为系统占用，不允许访问操作
* 程序中允许存在野指针，但是操作野指针可能报错
* 不建议将一个变量的值直接赋值给指针
* C语言中，可以把 NULL 赋值给指针，标志此指针为空指针，不指向任何空间

```c
int* p=NULL;
//此时无法对p进行读写操作


//NULL 是一个值为 0 的宏常量
#define NULL ((void*)0)
```

* 空指针是指内存地址编号为 0 的内存空间
* 操作空指针对应的空间一定会报错
* 空指针可以用作条件判断

#### 1.5、万能指针 void*

* void* 指针可以指向任意类型变量的内存空间

```c
void* p=NULL;

int a=10;
p=(void*)&a;	//指向变量时，最好转换为 void*

//使用指针变量指向的内存时，转换为 int*
*((int*)p)=11;
```

* 万能指针可以接收任意类型变量的内存地址
* 在通过万能指针修改变量的值的时候，需要将其转换为变量类型对应的指针类型才可以进行修改

#### 1.5、const 修饰的指针类型

```c
//常量
const int a=10;
a=100;		//在这里修改不了

int *p=&a;
*p=100;		//但是可以通过指针间接修改常量的值
```

1）const 修饰指针类型（只读指针）

* 可以修改指针变量 p 的值
* 不可以修改指针指向内存空间的值

```c
int  a=10;
int b=20;
// const 修饰指针类型
const int* p=&a;

p=&b;		//可以修改指针变量 p 的值

*p=100;		//err，不可以修改指针指向内存空间的值

```

2）const 修饰指针变量

* 不可以修改指针变量 p 的值
* 可以修改指针指向内存空间的值

```c
int  a=10;
int b=20;
//const 修饰指针变量的值
int* const p=&a;

p=&b;		//err，不可以修改指针变量 p 的值
*p=200;		//可以修改指针指向内存空间的值
```

3）const 修饰指针类型和指针变量

```c
int a=10;
int b=20;
//const 修饰指针类型和指针变量
const int* const p=&a;

p=&b;		//err，不可以修改指针变量 p 的值
*p=100；		//err，不可以修改指针指向内存空间的值
    
//二级指针的操作
int** pp=&p;
//*pp是一级指针的值
*pp=&b;		//这样就可以修改指针变量p的值了

**pp==100;		//这样就可以修改变量的值了（指针指向内存空间的值）
```

* 不可以修改指针变量 p 的值
* 不可以修改指针指向内存空间的值
* **但是，可以通过二级指针对指针变量 p 和指针指向内存空间的值进行修改**

### 2、指针和数组

#### 2.1、数组名

* 数组名是数组的首元素地址，是一个常量。

```c
//指针和数组
int arr[]={1,2,3,4,5,6,7,8,'a','b','c'};

arr=100;	//err，数组名是一个常量，不允许赋值

//数组名数数组首元素地址
int* p;		//指向数组的指针
p=arr;
printf("%p\n",p);
printf("%p\n",arr);

for(int i=0;i<10;i++)
{
    //下面几种写法等价，都可以达到访问数组元素的目的
    printf("%d\n",arr[i]);
    printf("%d\n",p[i]);
    printf("%p\n",*(arr+i));
    printf("%p\n",*(p+i));
    printf("%p\n",*p);
}

//两个指针相减，得到的结果是两个指针的偏移量
int step=p-arr;
printf("%d\n",step);		// 10

printf("%d\n",sizeof(p));	//4
printf("%d\n",sizeof(arr));		//40
```

#### 2.2、指针加减运算

* 指针变量 +1，等同于内存地址+sizeof(数据类型)
* 两个指针相减，得到的结果是两个指针的偏移量
* 所有类型的指针相减，结果都是整型数据，表示偏移的该类型的数据的个数
* 指向数组的指针操作数组：
    * 方法一：p[i]
    * 方法二：*(p+i)

* 指向数组的指针 p 和数组名的区别：
    * p 是变量，arr 是常量
    * 在上面的代码中：sizeof(p)=4，sizeof(arr)=40


**数组在作为函数参数时，会退化为指针，丢失了数组的精度（数组的元素个数信息）**

```c
void BubbleSort(int *arr,int len)
{
	for(int i=0;i<len-1;i++)
    {
        for(int j=0;j<len-1-i;j++)
        {
            if(*(arr+j)>*(arr+j+1))
            {
                int temp=*(arr+j);
                *(arr+j)=*(arr+j+1);
                *(arr+j+1)=temp;
            }
        }
    }
}
```

1）字符串拷贝：

```c
//下面几个 函数功能相同
void myStrCopy01(char* dest,char* ch)
{
    int i=0;
    while(ch[i])
    {
        dest[i]=ch[i];
        i++;
    }
    dest[i]=0;
}

void meStrCopy02(char* dest,char* ch)
{
    int i=0;
    while(*(ch+i))
    {
        *(dest+i)=*(ch+i);
        i++;
    }
    *(dest+i)=0;
}

void myStrCopy03(char* dest,char* ch)
{
    while(*ch)
    {
        *dest=*ch;
    	dest++;		//指针 +1，相当于指向数组下一个元素，内存地址变化了sizeof(数据类型)
    	ch++;
    }
    *dest=0;
}

void myStrCopy04(char* dest,char* ch)
{
    while(*dest++=*ch++);		
}

//字符串拷贝
char ch[]="hello world";
char dest[100];
myStrCopy(dest,ch);
printf("%s\n",dest);
```

2）指针加减：

```c
int arr[]={1,2,3,4,5,6,7,8,9,10};
int *p=arr;

arr[-1];	//err，数组下标越界
p=&arr[3];

//指针操作数组时，允许下标为负数
printf("%d\n",p[-2]);		//相当于是 *(p-2)

/*
p--;		//指针的加减运算和指针的类型有关
p--;
p--;		//此时p和arr代表的地址一样了
*/

int step=p-arr;		//step=3

```

3）指针和运算符的操作

* 乘、除、取余不能用于指针的运算
* 指针指针不可以直接相加
* 指针之间可以进行比较运算：>, >=, <, <=, ==, !=
* 指针可以进行逻辑判断：逻辑与、逻辑或

```c
int arr[]={1,2,3,4,5,6,7,8,9,10};
int *p=arr;

p=&arr[3];
if(p>arr)
{
    printf("真\n");
}

if(p&&arr)
{
    
}

p=p+arr;	//err，两个指针相加结果为野指针
p=p*arr;	//err
p=p*4;		//err
p=p/arr;	//err
```

### 3、指针数组

* 指针数组，是数组，数组的每个元素都是指针类型
* 数组指针，指向数组的指针（实际上应该没有这种提法）
* 指针数组对应于二级指针

```c
int a=10;
int b=20;
int c=30;

//定义指针数组
int* arr[3]={&a,&b,&c};
for(int i=0;i<sizeof(arr)/sizeof(arr[0]);i++)
{
    printf("%d\n",*arr[i]);
}

//sizeof(arr)	12
//sizeof(arr[0])	4

//指针数组里面的元素存储的都是指针
int a[]={1,2,3};
int b[]={4,5,6};
int c[]={7,8,9};

int* arr[]={a,b,c};

for(int i=0;i<3;i++)
{
    printf("%d\n",*arr[i]);		//输出的是：1	4	7
}

//下面三个语句的输出结果是一样的
printf("%p\n",arr[0]);
printf("%p\n",a);
printf("%p\n",&a[0]);


//指针数组是一个特殊的二维数组模型
printf("%d\n",arr[0][1]);		//输出的是a[1], 也就是2


for(int i=0;i<sizeof(arr)/sizeof(arr[0]);i++)
{
    for(int j=0;j<3;j++)
    {
        //二维数组形式打印出来
        printf("%d\n",arr[i][j]);
        
        //指针偏移量形式打印出来
        printf("%d\n",*(arr[i]+j));
        
        //arr是指针数组的首地址
        printf("%d\n",*(*(arr+i)+j));		//指针数组对应于二级指针
    }
}


```

### 4、多级指针

* 一级指针最常用
* 二级指针就是指向一个一级指针变量地址的指针
* 三级指针使用的比较少

```c
//指针数组和二级指针建立关系
int a[]={1,2,3};
int b[]={4,5,6};
int c[]={7,8,9};

int* arr[]={a,b,c};

int** p=arr;

printf("%d\n",**p);		//1
//二级指针加偏移量，相当于跳过了一个一维数组
printf("%d\n",**(p+1));		//4
//一级指针加偏移量，相当于跳过了一个元素
printf("%d\n",*(*p+1));		//2

printf("%d\n",*(*(p+1)+1));		//2

for(int i=0;i<3;i++)
{
    for(int j=0;j<3;j++)
    {
        //下面单个作用是一样的
        printf("%d ",p[i][j]);
        printf("%d ",*(p[i]+j));
        printf("%d ",*(*(p+i)+j));
    }
    puts("");
}


```

```c
int a=10;
int b=20;
int* p=&a;
int** pp=&p;
//pp 二级指针变量的值
//*pp 一级指针的值
//**pp 变量的值

// pp=100;		野指针
*pp=&b;		//相当于 p=&b，间接改变了 p 的值
printf("%d\n",*p);		//20

**pp=100;		//间接改变 b 的值
*pp=100;	//err 野指针

//三级指针
int*** ppp=&pp;		//*ppp==pp=&p		//**ppp==*pp==p==&a		***ppp==**pp==*p==a



```

### 5、指针和函数

#### 5.1、值传递和引用传递

1）值传递

```c
void swap(int a,int b)
{
    int temp=a;
    a=b;
    b=temp;
}

int a=10;
int b=20;
//值传递，形参不影响喜实参的值
swap(a,b);
printf("%d %d\n",a,b);		//10 20

```

2）指针作为函数参数（地址传递）

```c
//指针作为函数参数
void swap(int* a,int* b)
{
    int temp=*a;
    *a=*b;
    *b=temp;
}

int a=10;
int b=20;
//地址传递，形参可以改变实参的值
swap(&a,&b);
printf("%d %d\n",a,b);		//20 10

```

#### 5.2、数组名作为函数参数

* 数组名作函数参数，函数的形参会退化为指针

1）数组名作函数参数（数组写法）

```c
void myStrCat(char* ch1,char* ch2)
{
    int i=0;
    while(ch1[i++]);		//相当于i=strlen(ch1)
    int j=0;
    while(ch2[j])
    {
        ch1[i+j]=ch2[j];
        j++;
    }
}

char ch1[100]="Hello";
char ch2[]="world";
myStrCat(ch1,ch2);

```

2）数组名作函数参数（指针写法）

```c
void myStrCat(char* ch1,char* ch2)
{
    int i=0;
    while(*(ch1+i)!='\0')		//相当于i=strlen(ch1)
    {
        i++
    }
    int j=0;
    while(*(ch2+j))
    {
        *(ch1+i+j)=*(ch2+j);
        j++;
    }
}

char ch1[100]="Hello";
char ch2[]="world";
myStrCat(ch1,ch2);
```

3）数组名作函数参数（指针写法）

```c
void myStrCat(char* ch1,char* ch2)
{
    while(*ch1)		//相当于i=strlen(ch1)
    {
        ch1++;
    }
    while(ch2)
    {
       *ch1=*ch2;
        ch1++;
        ch2++;
    }
}

//myStrCat1作用和myStrCat一样
void myStrCat1(char* ch1,char* ch2)
{
    while(*ch1) ch1++;
    while(*ch1++=*ch2++);
}

char ch1[100]="Hello";
char ch2[]="world";
myStrCat(ch1,ch2);
```

#### 5.3、字符串去空格

1）使用辅助空间

```c
//去字符串空格
void removeSpace(char* ch)
{
    char str[100]={0};
    char* temp=str;
    int i=0;
    int j=0;
    while(ch[i])
    {
        if(ch[i]!=' ')
        {
            str[j]=str[i];
            j++
        }
        i++;
    }
    while(*ch++==*temp++);
}

char ch[]="h  e ll o   w";
removeSpace(ch);
```

2）不使用辅助空间

```c
//去字符串空格
void removeSpace(char* ch)
{
    int i=0;
    int j=0;
    while(ch[i])
    {
        if(ch[i]!=' ')
        {
            ch[j]=ch[i];
            j++;
        }
        i++;
    }
    ch[j]=0;
}

char ch[]="h  e ll o   w";
removeSpace(ch);
```

#### 5.4、指针作为函数的返回值

1）数组形式

```c
char* myChr(char* str,char ch)
{
    int i=0;
    while(str[i])
    {
        if(str[i]==ch)
        {
            return &str[i];
        }
        i++;
    }
    return NULL;
}

char ch[]="hello world";
char* res=myChr(ch,l);
if(!res)
    printf("未找到\n");
```

2）指针形式

```c
char* myChr(char* str,char ch)
{
    while(*str)
    {
        if(*str==ch)
        {
            return &str;
        }
        str++;
    }
    return NULL;
}

char ch[]="hello world";
char* res=myChr(ch,l);
if(!res)
    printf("未找到\n");
else
    printf("%s\n",res);
```

#### 5.5、字符串查找字符串

```c
char* myStrStr(char* src,char* dest)
{
    char* fsrc=src;		//遍历源字符串的指针
    char* rsrc=src;		//记录每次相同字符串首地址
    char* tdest=dest;
    while(*fsrc)
    {
        rsrc=fsrc;
        while(*fsrc=*tdest&&*fsrc!='\0')
        {
            fsrc++;
            tdest++;
        }
        if(*tdest=='\0')
            return rsrc;
        else		//回滚
        {
            //目标字符串回滚
            tdest=dest;
            fsrc=rsrc;
            fsrc++;
        }
    }
    return NULL;
}

char src[]="hello world";
char dest[]="llo";
char* res=myChr(src,dest);
if(res==NULL)
{
    printf("未找到\n");
}
else
{
    printf("%s\n",res);
}
```

### 6、指针和字符串

#### 6.1、字符指针

1）常量字符串

```c
char ch[]="hello world";
char* p=ch;
printf("%s\n",p);

// char* p="hello world";		也是合法的，可以把p当做字符数组使用
/* 但是他们存在不同之处：
1)字符数组可以修改数组元素
2）无法通过 p[i]='m' 这样的方式修改内容
*/

char* p="hello world";
char* p1="hello world";		//p 和 p1 是相同的地址

```

* **char* p="hello world";		也是合法的，可以把p当做字符数组使用**
    但是他们存在不同之处：
    1）字符数组可以修改数组元素
    2）无法通过 p[i]='m' 这样的方式修改内容

    3）char ch[]="hello world"，是栈区字符串

    ​	char* p="hello world"，是数据区常量字符串，只读的

2）字符串数组

```c
//字符串数组
//指针数组
char ch1[]="hello";
char ch2[]="world";
char ch3[]="dabaobeier";
char* arr1[]={ch1,ch2,ch3};		

char* arr2[]={"hello","world","dabaobeier"};		//字符串数组，内容不可修改
for(int i=0;i<3;i++)
{
    printf("%s\n",arr2[i]);
}

//字符串排序
for(int i=0;i<3-1;i++)
{
    for(int j=0;j<3-1-i;j++)
    {
        if(arr2[j][0]>arr[j+1][0])		//首字符进行比较
        {
            //交换字符串
            char* temp=arr2[j];
            arr2[j]=arr2[j+1];
            arr2[j+1]=temp;
        }
    }
}

for(int i=0;i<3;i++)
{
    printf("%s\n",arr2[i]);
}
```

#### 6.2、字符指针作为函数参数

* 当作为函数的指针所指向的内容在函数内不需要被修改时，可以使用 const 修饰指针参数：

    **int myStrLen(const char* ch)**

```c
//数组实现
int myStrLen(char* ch)
{
    int i=0;
    while(ch[i])
        i++；
    return i;
}

//指针实现
int myStrLen(const char* ch)	//在函数中，不希望指针指向的内容被修改，若被修改会报错，从而保护不希望被修改的参数
{
    char* temp=ch;
    while(*temp)	temp++;
    return temp-ch;
}

char ch[]="hello world";
int lengthCh=myStrLen(ch);		//11
```

#### 6.3、const 修饰的指针变量

* **const 修饰的总是离它最近的内容**

1）通过一级指针修改变量的值

```c
const int a=10;
a=100;		//err
int* p=&a;
*p=100;		//通过一级指针修改变量的值
```

2）指向常量的指针

* 指向常量的指针，可以修改指针变量的值，不能修改指针变量指向内存空间的值

```c
char ch1[]="hello";
char ch2[]="world";

const char* p=ch1;

*p='m';		//err
p[2]='m';	//err，等同于 *(p+2)=m
p=ch2;		//没问题

```

3）常量指针

* 常量指针，可以修改指针变量指向内存空间的值，不可以修改针变量的值

```c
char ch1[]="hello";
char ch2[]="world";

//常量指针
char* const p=ch1;

p=ch2;	//err，常量指针不可以修改
p[2]='m';	//没问题
*(p+2)='m';		//没问题
```

4）只读指针

* 只读指针，不可以修改指针变量指向内存空间的值，不可以修改针变量的值
* 只读指针，可以通过二级指针修改只读指针指向的内存空间的内容
* 只读指针，可以通过二级指针修改只读指针指向的内存空间
* 由此可见，通过 const 修饰也并不是安全的，可以通过高级指针进行修改

```c
char ch1[]="hello";
char ch2[]="world";

const char* const p=ch1;
p=ch2;		//err
*p='m';		//err
p[2]="m";	//err

char** pp=&p;
*(*pp+1)='m';	//可以通过二级指针修改只读指针指向的内存空间的内容
*pp=ch2;	//可以通过二级指针修改只读指针指向的内存空间

//由此可见，通过 const 修饰也并不是安全的，可以通过高级指针进行修改
```

#### 6.4、主函数参数

```c
int main(int argc,char* argv[]);
```

* main 函数是由操作系统调用的，第一个参数 argc 标明 argv 数组的成员数量，argv 数组的每个成员都是 char* 类型；
* argv 是命令行参数的字符串数组
* argc 代表命令行参数的数量，程序名字本身算一个参数。

```c
//主函数的形参
//gcc -o hello.out hello.c	4个参数：gcc，-o,hello.out，hello.c 
//int argc	传递参数的个数
//char* argv[]	参数：char* argv[]={"gcc","-o","hello.out","hello.c"}，表示参数的具体内容
int main(int argc,char* argv[])
{
    //参数个数判断
    if(argc<3)
    {
        printf("缺少参数\n");	//在命令行运行程序时，若参数个数不一致，会报错
        return -1;
    }
	for(int i=0;i<argc;i++)
    {
        printf("%s\n",argv[i]);			
    }
    return 0;
}
```

#### 6.5、字符串中某个字符串出现的次数

1）while 模型

```c
char* myStrStr(char* src,char* dest)
{
    char* fsrc=src;		//遍历源字符串的指针
    char* rsrc=src;		//记录每次相同字符串首地址
    char* tdest=dest;
    while(*fsrc)
    {
        rsrc=fsrc;
        while(*fsrc=*tdest&&*fsrc!='\0')
        {
            fsrc++;
            tdest++;
        }
        if(*tdest=='\0')
            return rsrc;
        else		//回滚
        {
            //目标字符串回滚
            tdest=dest;
            fsrc=rsrc;
            fsrc++;
        }
    }
    return NULL;
}

char *str="hello world";
char ch[]="llo";
char* p=myStrStr(str,ch);
int cnt=0;		//记录出现的次数
while(p)
{
    cut++;
    p=p+strlen(ch);
    p=myStrStr(p,ch);
}

printf("%s 在 %s 中出现的次数：%d\n",ch,str,cnt);

```

2）do-while 模型

```c
char* myStrStr(char* src,char* dest)
{
    char* fsrc=src;		//遍历源字符串的指针
    char* rsrc=src;		//记录每次相同字符串首地址
    char* tdest=dest;
    while(*fsrc)
    {
        rsrc=fsrc;
        while(*fsrc=*tdest&&*fsrc!='\0')
        {
            fsrc++;
            tdest++;
        }
        if(*tdest=='\0')
            return rsrc;
        else		//回滚
        {
            //目标字符串回滚
            tdest=dest;
            fsrc=rsrc;
            fsrc++;
        }
    }
    return NULL;
}

char *str="hello world";
char ch[]="llo";
char* p=myStrStr(str,ch);
int cnt=0;		//记录出现的次数
do
{
    if(p)
    {
        cut++;
        p+=strlen(ch);
        p=myStrStr(p,ch);
    }
}while(p);

printf("%s 在 %s 中出现的次数：%d\n",ch,str,cnt);

```

#### 6.6、统计非空格字符个数

1）数组格式

```c
int getStrCnt(char* ch)
{
    int i=0;
    int cnt=0;
    while(ch[i])
    {
        if(ch[i]!=' ')
            cnt++;
        i++;
    }
    return cnt;
}

char ch[]="h  e l  lo  wo   rld ";
int res=getStrCnt(ch);

```

2）指针格式

```c
int getStrCnt(char* ch)
{
    int cnt=0;
    while(*ch)
    {
        if(*ch!=' ')
            cnt++;
        ch++;
    }
    return cnt;
}

char ch[]="h  e l  lo  wo   rld ";
int res=getStrCnt(ch);

```

3）统计各个字符出现的次数

```c
//hash表
char ch[]="h  e l  lo  wo   rld nichousha chounizadi zaichuyigeshishi";
int hash[27]={0};
while(*ch)
{
    if(*ch==' ')
    {
        hash[26]++;
    }
    else
    {
        hash[*ch-'a']++;
    }
    ch++;
}

printf("空格出现了 %d 次\n",hash[26]);
for(int i=0;i<26;i++)
{
    printf("字符 %c 出现的次数为 %d\n",i+'a',hash[i]);
}

```

### 7、常用的字符串操作

#### 7.1、字符串逆置

1）数组形式

```c
void inverse(char* ch)
{
    int i=0;
    int j=strlen(ch)-1;
    while(i<j)
    {
        char temp=ch[i];
        ch[i]=ch[j];
        ch[j]=temp;
        i++;
        j--;
    }
}


//字符串逆置
char ch[]="hello world";
inverse(ch);

```

2）指针形式

```c
void inverse(char* ch)
{
    char* ftemp=ch;
    char* btemp=ch+srtlen(ch)-1;
    while(ftemp<btemp)
    {
        char temp=*ftemp;
        *ftemp=*btemp;
        *btemp=*ftemp;
        ftemp++;
        btemp--;
    }
}


//字符串逆置
char ch[]="hello world";
inverse(ch);

```

#### 7.2、字符串拷贝

1）strcopy()

```c
#include<string.h>
char* strcopy(char* dest,const char* rsc);
```

* 功能：把 src 所指向的字符串复制到 dest 所指向的空间，'\0' 也会拷贝过去
* 参数：
    * dest：目的字符串首地址
    * src：源字符串首地址
* 返回值：
    * 成功：返回 dest 字符串的首地址
    * 失败：NULL

**注：如果 dest 所指向的内存空间不够大，可能会造成缓冲溢出的错误** 

```c
#include<string.h>
char ch[]="hello world";
char str[100];
strcopy(str,ch);
```

strcpy()

```c
void myStrCpy(char* dest,const char* src)
{
	while(*dest++=*src++);
}
```

2）strncopy()

```c
#include<string.h>
char* strncopy(char* dest,const char* src,size_t n);
```

* 功能：把 src 所指向的字符串的前 n 个字符复制到 dest 所指向的空间，'是否拷贝结束符取决于指定的长度是否包含结束符
* 参数：
    * dest：目的字符串首地址
    * src：源字符串首地址
    * n：指定需要拷贝字符个数
* 返回值：
    * 成功：返回 dest 字符串的首地址
    * 失败：NULL

**注：当指定长度不包括字符串结束符时， dest 内不包含结束符，如果直接打印 dest 会出现越界现象** 

```c
//字符串有限拷贝
#include<string.h>
char ch[]="hello world";
char str[100];
strcopy(str,ch，5);
```



#### 7.3、字符串追加

1）strcat()

```c
#include<string.h>
char* strcat(char* dest, const char* src);
```

* 功能：把 src 字符串连接到 dest 的尾部，'\0'也会追加过去
* 参数：
    * dest：目的字符串首地址
    * src：源字符串首地址
* 返回值：
    * 成功：返回 dest 字符串的首地址
    * 失败：返回 NULL

**注：如果 dest 所指向的内存空间不够大，可能会造成缓冲溢出的错误** 

```c
#include<string.h>
char* ch[]="hello";
char* src[]="world";

strcat(ch,src);
```

```c
void myStrCat(char* dest,const char* src)
{
    while(*dest++);
    while(*dest++=*src++);
}
```



2）strncat()

```c
#include<string.h>
char* strcat(char* dest, const char* src,size_t n);
```

* 功能：把 src 字符串连接到 dest 的尾部，'\0'也会追加过去
* 参数：
    * dest：目的字符串首地址
    * src：源字符串首地址
    * n：指定需要追加字符的个数
* 返回值：
    * 成功：返回 dest 字符串的首地址
    * 失败：返回 NULL

**注：如果 dest 所指向的内存空间不够大，可能会造成缓冲溢出的错误** 

**如果指定长度大于源字符串的长度，会将整个源字符串追加过去，不会报错**

```c
#include<string.h>
char* ch[]="hello";
char* src[]="world";

strncat(ch,src,n);
```

```c
void myStrCat(char* dest,const char* src)
{
    while(*dest++);
    while(*dest++=*src++&&n--);
}
```



#### 7.4、字符串比较

1）strcmp()

```c
#include<string.h>
int strcmp(const char* s1,const char* s2);
```

* 功能：比较 s1 和 s2 的大小，比较的是字符 ASCII 码的大小
* 参数：
    * s1：字符串1的首地址
    * s2：字符串2的首地址
* 返回值：
    * 相等：0
    * 大于：>0
    * 小于：<0

**不相等的时候，有的操作系统返回 ASCII 码的差值，有的操作系统返回1 或者 -1** 

```c
//字符串比较
#include<string.h>
char ch1[]="hello";
char ch2[]="world";

int value=strcmp(ch1,ch2);
```

2）strncmp()

```c
#include<string.h>
int strncmp(const char* s1,const char* s2,size_t n);
```

* 功能：比较 s1 和 s2 前 n 个字符的大小，比较的是字符 ASCII 码的大小
* 参数：
    * s1：字符串1的首地址
    * s2：字符串2的首地址
    * n：指定比较字符的数量
* 返回值：
    * 相等：0
    * 大于：>0
    * 小于：<0

**不相等的时候，有的操作系统返回 ASCII 码的差值，有的操作系统返回1 或者 -1** 

```c
//字符串比较
#include<string.h>
char ch1[]="hello";
char ch2[]="world";

int value=strcmp(ch1,ch2,5);
```

```c
int myStrCmp(const char* s1,const char* s2)
{
    while(*s1==*s2)
    {
        if(!*s1)
            return 0;
        s1++;
        s2++;
    }
    return *s1>*s2?1:-1;
}
```

```c
//这个版本不如上面的版本明朗
int myStrCmp(const char* s1,const char* s2)
{
    while(*s1++==*s2++ && !*s1)
    {
        if(!*--s1 && !*--s2)
            return 0;
    }
    return *s1>*s2?1:-1;
}
```

```c
int myNStrCmp(const char* s1,const char* s2,size_t n)
{
    for(int i=0;i<n && s1[i] && s2[i];i++)
    {
        if(s1[i]!=s2[i])
        {
            return s1[i]>s2[i]?1:-1;
        }
    }
}
```



#### 7.5、字符串格式化

1）sprintf() ：格式化输出

```c
#include<stdio.h>
int sprintf(char* str,const char* format,...);
```

* 功能：根据参数 format 来转换并格式化数据，然后将结果 str 输出到指定的空间中，直到出现字符串结束符为止。**实际就是将format 的内容存放到 str 中，并输出** 
* 参数：
    * str：字符串首地址
    * format：字符串格式，用法与 printf 相同
* 返回值：
    * 成功：实际格式化的字符个数
    * 失败：-1

**注：format 也可以使用各种格式控制** 

```c
#include<stdio.h>
char ch[100];
char ch2[100];
sprintf(ch，"hello world");		//输出：hello world
printf("%s\n",ch);				//输出：hello world

//format 也可以使用各种格式控制
sprintf(ch2,"%d+%d=%d\n",1,2,3);	//输出：1+1=2
printf("%s\n",ch1);					//输出：1+1=2
```

2）sscanf()：格式化读入

```c
#include<stdio.h>
int sscanf(const char* str,const char* format,...);
```

* 功能：从 str 指定的字符串读取数据，并根据参数 format 字符串的格式来进行格式转换并格式化数据。**实际就是将 str 按照 format 的格式进行格式转化** 
* 参数：
    * str：指定的字符串首地址
    * format：字符串样式，用法和 sacnf 相同
* 返回值：
    * 成功：参数数目，成功转换的值的个数
    * 失败：-1

```c
char ch[]="1+2=3";
int a,b,c;
sscanf(ch,"%d+%d=%d",&a,&b,&c);		//把 ch 按照 "%d+%d=%d",&a,&b,&c 的格式分别存入 a、b、c 中

printf("%d\n",a);		//1
printf("%d\n",b);		//2
printf("%d\n",c);		//3
```

```c
char ch[]="hello world";
char str[100];
char str1[100];

sscanf(ch,"%s",str);	//把 ch 中的内容按照字符串的格式读取到 ch 中，实际读取到的是 hello
printf("%s\n",str);		//输出：hello

sscanf(ch,"%[^\n]s",str1);	//把 ch 中的内容按照字符串的格式读取到 ch 中，实际读取到的是 hello world
printf("%s\n",str1);		//输出：hello world

char ch1[]="helloworld";
char str2[100];

sscanf(ch1,"%5s",str2);	//把 ch 中的前五个字符按照字符串的格式读取到 ch 中，实际读取到的是 hello
printf("%s\n",str2);		//输出：hello

char ch1[]="helloworld";
char str2[100];
char str3[100];

sscanf(ch1,"%5s%s",str2,str3);	
printf("%s\n",str2);		//输出：hello
printf("%s\n",str3);		//输出：world
```



#### 7.6、字符串查找

1）strchr()

```c
#include<string.h>
char* strchr(const char* s,int c);
```

* 功能：在字符串 s 中查找字符 c 出现的位置
* 参数：
    * s：字符串首地址
    * c：匹配字符
* 返回值：
    * 成功：返回第一次出现 c 的地址
    * 失败：NULL

```c
char* strchr(const char* ch,int c)
{
    while(*ch)
    {
        if(*ch==c)
            return ch;
        ch++;
    }
    return NULL;
}

char ch[]="hello world";
char c='l';
char* p=strchr(ch,c);

printf("%s\n",p);
```



2）strstr()

```c
#include<string.h>
char* strstr(const char* haystack,const char* needle);
```

* 功能：在字符串 haystack 中查找字符串 needle 出现的位置
* 参数：
    * haystack：源字符串的首地址
    * needle：匹配字符串的地址
* 返回值：
    * 成功：返回第一次出现 needle 的地址
    * 失败：NULL

```c
char ch[]="hello world";
char str[]="llo";
char* p=strstr(ch,str);
printf("%s\n",p);
```



#### 7.7、字符串分割

1）strtok()

```c
#include<string.h>
char* strtok(char* str,const char* delim);
```

* 功能：将字符串分割成一个个片段，当 strtok() 在参数 str 的字符串中发现参数 delim 中包含的分割字符时，则会将该字符改为 0 字符，当连续出现多个时，只替换第一个为 0
* 参数：
    * str：指向欲分割的字符串
    * delim：为分割字符串中包含的所有字符
* 返回值：
    * 成功：分割后字符串首地址
    * 失败：NULL

**在第一次调用时，strtok() 必须基于参数 str 字符串**

**在往后的调用，则将参数 str 设置成 NULL，每次调用成功则返回指向被分割出片段的指针** 

```c
char ch[]="www.baidu.com";

//第一次截取
char* p=strtok(ch,".");
printf("%s\n",p);		//输出：www
printf("%s\n",ch);		//输出：www	此时 ch 中的内容为：www\0baidu.com

printf("%p\n",p);		//输出：0019F7E4
printf("%p\n",ch);		//输出：0019F7E4

//第二次截取
p=strtok(NULL,".");
printf("%s\n",p);		//输出：baidu	此时，ch 中的内容为：www\0baidu\0com

//第三次截取
p=strtok(NULL,".");
printf("%s\n",p);		//输出：com	
```

* strtok() 会破坏源字符串 str，用 \0 替换分割的标志位

```c
char ch[]="123456789@qq.com";
char chCopy[100]={0};

//字符串备份
strcopy(chCopy,ch);

char* p=strtok(str,"@");
printf("%s\n",p);		//输出：123456789

p=strtok(NULL,".");
printf("%s\n",p);		//输出：qq


```

```c
char* ch[]="nichousha\nchounizadi\nzaichouyixiashishi\nduibuqidagewocuole\nguawazi\n";
char*p=strtok(ch,"\n");
printf("%s\n",p);		//输出：nichousha

//利用循环多次截取
while(p)
{
    printf("%s\n",p);
    p=strtok(NULL,"\n");
}
/*
输出：
nichousha
chounizadi
zaichouyixiashishi
duibuqidagewocuole
guawazi
*/
```

#### 7.8、字符串类型转换

1）atoi()

```c
#include<stdlib.h>
int atoi(const char* nptr);
```

* 功能：atoi() 会扫描字符串，跳过前面的空格字符，直到遇到数字或者正负号才开始做转换，而遇到非数字或字符串结束符才结束转换，并将结果返回
* 参数：nptr，待转换的字符串
* 返回值：成功转换后的整数

**类似的函数：**

* atof()：把一个小数形式的字符串转换为一个浮点数（特指双精度浮点型）
* atol()：将一个字符串转换成 long 类型

```c
#include<stdlib.h>

/*
char ch[]="123456";

int i=atoi(ch);
printf("%d\n",i);		//输出：123456
*/

/*
char ch[]="-123456";

int i=atoi(ch);
printf("%d\n",i);		//输出：-123456
*/

/*
char ch[]="    -123456abc123";

int i=atoi(ch);
printf("%d\n",i);		//输出：-123456
*/

char ch[]="  asc   -123456abc123";

int i=atoi(ch);
printf("%d\n",i);		//输出：0


```

**注：只能识别十进制**

```c
char ch[]="   -123.456abc123";

int i=atof(ch);
printf("%f\n",i);		//输出：-123.456
```

```c
char ch[]="   -123.456abc123";

int i=atof(ch);
printf("%d\n",i);		//输出：-123
```