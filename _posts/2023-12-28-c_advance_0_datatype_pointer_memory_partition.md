---
layout: post
title: c_advance_0_datatype_pointer_memory_partition
date: 2023-12-28
---

## 数据类型-变量-内存四区-指针

### 1、内存四区

#### 1.1、数据类型的本质

1）数据类型基本概念

* 类型是对数据的抽象
* 类型相同的数据具有相同的表示形式、存储格式、相关的操作
* 程序中使用的数据必定属于某种数据类型
* 数据类型和内存 有关系
* C/C++ 引入数据类型，可以更方便地管理数据

2）数据类型的本质

* 数据类型可以理解为创建变量的模具：固定内存大小的别名
* 数据类型的作用：编译器预算对象（变量）分配的内存空间大小
* 数据类型只是模具，编译器不为类型分配空间，只有根据类型创建的变量才会分配空间

```c
int main()
{
    int a;			//告诉编译器分配 4 个字节
    int b[10];		//告诉编译器分配 4*10 个字节
    //类型的本质是固定大小内存的别名
    
    printf("sizeof(a)=%d\nsizeof(b)=%d\n",sizeof(a),sizeof(b));		//4 40
    
    //打印地址
    printf("b:%d	&b:%d\n",b,&b);		//数组名字就是数组首元素地址，数组首地址		两个输出一样的
    
    
    //b 和 &b 的数据类型不一样
    //b，数组首元素地址，一个元素 4 字节，+1 -> +4
    //&b，整个数组的首地址，一个数组 4*10=40字节，+1 -> +40
    printf("b+1:%d	&b+1:%d\n",b+1,&b+1);		//14678444 14678480
    
    //指针类型长度，32位为4, 64位为8
    char *****************p=NULL;
    int *q=NULL;
    printf("%d %d\n",sizeof(p),sizeof(q));		//32位：4 4		64位：8 8
    
    return 0;
}
```

3）数据类型的别名

* typedef 可以给类型取别名，且 typedef 只能给类型取别名

```c
typedef unsigned int u32;

//typedef 通常和结构体一起使用
typedef struct	myStruct		//这里的 myStruct 可写可不写
{
    int a;
    int b;
}TMP;
int mian()
{
    u32 t;				//unsigned int t;
    return 0;
}
```

4）void 类型（空类型、无类型）

* **函数参数为空**，在定义函数的时候，可以使用 void 来修饰：int func(void);		在 C++ 中 void 写不写是一样的，但是在 C 中是存在区别的
* **函数没有返回值**，使用 void 来修饰：void func(void);
* **不能定义 void 类型的普通变量**：void a;        //err
* **可以定义 void* 类型的指针**：void* p;
    * 主要是因为 void 类型的普通变量不同类型的内存大小不一样，在编译器分配内存空间的时候无法确定分配的内存大小，而在相同的操作系统下，不同类型的指针变量内存大小相同，不影响编译器的内存分配
* **void* p 万能指针**，常用作函数返回值，或者函数的参数
    * 这样可以很灵活，只要是指针就可以使用，例如 molloc 函数的定义：void* molloc(size_t size)
    * memcpy 函数，拷贝内存的内容，可以拷贝各种类型的数组，而 strcpy，只能拷贝 char 类型的数组

#### 1.2、变量的使用

* C 语言中，一维数组、二维数组其实也是有数据类型的
* C 语言中，函数也是具有数据类型的，可以通过函数指针进行重定义

1）变量的本质

* 变量：既能度又能写的内存对象。一旦初始化后不能修改的称为称量
* 变量定义形式：
    * 类型 标识符1，标识符2，…，标识符n
* 变量的本质是：**一段连续内存空间的别名**

```c
int main()
{
    int a;
    
    //变量相当于门牌号，内存相当于房间
    //直接赋值
    a=10;
    
    pringf("a=%d\n",a);		//10
    
    //间接赋值
    pringf("&a=%p\n",&a);		// a的地址
    p=&a;
    pringf("p=%p\n",p);		// a的地址
    
    *p=22;
    pringf("a=%d\n",a);		//22
    pringf("*p=%d\n",*p);		//22
    
    return 0;
}
```

#### 1.3、内存四区模型

* 四区：栈区、堆区、全局区、代码区（不用管）

1）全局区（静态区）：全局变量、静态变量、文字常量

* 全局变量和静态变量，初始化的全局变量和静态变量存储在一起，未初始化的全局变量在另一块
* 全局区相同的常量只存在一份

```c
char *get_str1()
{
    char *p="abcdef";		//文字常量区
    return p;
}

char *get_str1()
{
    char *q="abcdef";		//文字常量区
    return q;
}

int main()
{
    char *p=NULL;
    char *q=NULL;
    
    p=get_str1();
    
    //%s，打印指针指向的内存区域的内容
    //%d，打印 p 本身的值
    printf("p=%s, p %d\n",p,p);			//p=abcdef,p=一串地址
    
    q=get_str2();
    printf("q=%s, q %d\n",q,q);			//q=abcdef,q=一串地址，且这个地址和p的地址一样
    
    //主函数里面的p和 get_str1里面的p对应不同的内存
    //全局区相同的常量只存在一份，因此主函数里面的 p 和 q 指向同一块内存
    
    return 0;
}
```

2）栈区

```c
char* get_str()
{								//字符串 "sdskcnckjana" 是存放在全局区
    char str[]="sdskcnckjana";		//栈区，函数结束，内存销毁，主函数中复制内存的内容，因此复制到的内容是不确定的，可能是原本的内容，也可能是乱码
    							
    							//这里在 char str[]="sdskcnckjana" 之后，会拷贝一份字符串到栈区
    return str;
}

int main()
{
    char* buffer[128]={0};
    
    strcpy(buffer,get_str());
    printf("%s\n",buffer);		//打印的结果：不确定，即乱码,这里还有可能输出 sdskcnckjana，是因为这里拷贝的时候可能 get_char 还没有销毁
    return 0;
}
```

```c
char* get_str()
{								//字符串 "sdskcnckjana" 是存放在全局区
    char str[]="sdskcnckjana";		//栈区，函数结束，内存销毁，主函数中复制内存的内容，因此复制到的内容是不确定的，可能是原本的内容，也可能是乱码
    printf("%s\n",buffer);		//打印的结果：sdskcnckjana
    							
    							//这里在 char str[]="sdskcnckjana" 之后，会拷贝一份字符串到栈区
    return str;
}

int main()
{
    char* buffer[128]={0};
    
    char* p=NULL;
    p=get_str();
    printf("%s\n",buffer);		//打印的结果：不确定，即乱码
    
    return 0;
}
```

3）堆区

```c
char* get_str()
{								//字符串 "sdskcnckjana" 是存放在全局区
    char str[]="sdskcnckjana";		//栈区，函数结束，内存销毁，主函数中复制内存的内容，因此复制到的内容是不确定的，可能是原本的内容，也可能是乱码
    printf("%s\n",buffer);		//打印的结果：sdskcnckjana
    							
    							//这里在 char str[]="sdskcnckjana" 之后，会拷贝一份字符串到栈区
    return str;
}

char* strget2()
{
    char *temp=(char*)malloc(100);		//堆区分配空间
    
    if(temp==NULL)
        return NULL;
    strcpy(tmp,"snjcscsdmkcs");
    //在这里，字符串"snjcscsdmkcs"存放在全局区，temp存放在栈区，指向一块堆区的内存，strcpy之后，会拷贝一份字符串"snjcscsdmkcs"到temp指向的堆区内存
    //get_str2函数运行完毕之后，p也会指向temp指向的堆区内存，并且会释放指针temp，但是temp指向的内存不会释放，需要手动释放
    return temp;
}
int main()
{
    char* buffer[128]={0};
    
    char* p=NULL;
    p=get_str2();
    if(!p)
    {
        printf("%s\n",buffer);		//打印的结果：snjcscsdmkcs
        free(p);		//释放p之前，这块堆内存使用权归p，释放之后，使用权归操作系统，但是内部的内容依然存在，直到下次被写才会改变，并且释放p之后，p依然指向这块堆区域，只是p指向的堆内存可以由系统支配了，所以一般指针释放之后，会将其指向空指针
        p=NULL;
    }
    	
    
    return 0;
}
```

#### 1.4、函数调用模型

* 关注重点在于调用的流程和变量的生命周期
* 调用模型是一个栈模型：先调用，后返回

![image-20220531161154720](E:\Notes\C\image-20220531161154720.png)

#### 1.5、函数调用变量传递分析

* main 函数调用子函数1，子函数1调用子函数2，那么 main 函数在栈区开辟的内存，子函数1和子函数2都可以使用
* main 函数在堆区开辟的内存，没有释放的时候，子函数1和子函数2都可以使用
* 子函数1在栈区开辟的内存，子函数1和子函数2都可以使用，但是 main 函数无法使用
* 子函数在堆区开辟的内存，没有释放的时候，main 函数、子函数1和子函数2都可以使用
* 全局区存放的变量，生命周期和程序一致，因此无论哪个函数在全局区开辟的内存，所有函数都可以使用

#### 1.6、静态局部变量的使用

```c
int *getA()
{
    static int a=10;	//a是一个局部的静态变量，函数结束，内存不释放，因此只要把地址传出去，就可以通过地址使用这个内存了
    return &a;
}

int main()
{
    int *p=getA();		//通过地址使用局部静态变量
    
    return 0;
}
```

* 在变量的生命周期之外，只要内存没有释放，就能够通过一定的手段使用对应的内存

#### 1.7、栈的生长方向和内存释放方向

* 栈底，高地址；栈顶，低地址。栈的生长方向：栈底到栈顶，即栈的高地址到低地址，一般描述为从上到下
* 堆的生长方向与栈相反，从低地址到高地址，一般描述为从下到上
* 栈内的数组内部，是从低地址向高地址的，即栈内数组内部，也是从下到上

### 2、指针强化

#### 2.1、指针也是一种数据类型

* 指针变量也是一种变量，占有内存空间，用来保存内存地址
* 通过星号操作内存
    * 在指针声明的时候，星号表示所声明的是指针变量
    * 在指针使用的时候，星号表示操作指针所指向的内存空间中的值
    * *p 相当于通过地址（p 变量的值）找到一块内存，然后操作内存

```c
int main()
{
    int a=100;
    int *p=NULL;
    int p1=NULL;
    char *********q=0x1111;
    
    //指针指向谁，就把谁的地址赋值给指针
    p1=&a;
    //通过星号可以找到指针指向的内存区域，操作的还是内存
    //星号放在等号的左边，给内存赋值，写内存
    //星号放在等号的右边，取内存赋值，读内存
    *p1=22;
    
    printf("%d %d\n",sizeof(p),sizeof(q));		//32位系统：4 4
    return 0;
}
```

#### 2.2、指针间接赋值

* void* 类型的指针在使用的时候需要转换成实际的类型

```c
int main()
{
    void* p;
    char buf[1024]="aancnciwce";
    p=buf;
    //void* 类型的指针在使用的时候需要转换成实际的类型
    printf("%s\n",(char*)p);
    
    int a[100]={1,2,3,4};
    p=a;
    int i=0;
    //void* 类型的指针在使用的时候需要转换成实际的类型
    for(i=0;i<4;i++)
    {
        printf("%d ",*((int*)p+i));
    }
    
    int b[3]={1,2,3};
    int c[3];
    memcpy(c,b,sizeof(b));		//void* 类型的指针转换成了 int*
    for(i=0;i<3;i++)
    {
        printf("%d ",c[i]);
    }
    
    char *q=NULL;			//#define NULL ((void*)0)，这里实际上q没有具体的指向，所以给q指向的内存赋值就会报错
    
    /*
    //加上这样两句，就可以给q 一个具体的指向，再给其指向的内存赋值，就没问题了
    char str2[100]={0};
    q=str2;
    */
    
    //给 q 指向的内存区域赋值
    strcpy(q,"1234");		//err
    
    return 0;
}
```

* 分文件编程说明

```c
//防止头文件重复包含
#pragma once

//例如，有两个头文件 a.h 和 b.h，且在 a.h 中包含了 b.h，在 b.h 中包含了 a.h，那么会出现头文件包含的死循环，导致文件包含过多的错误

//不添加兼容 C++，使用 C++ 语言的程序调用 C 语言的程序的语句，编译的时候不会有问题，但是使用的时候会有问题。添加了之后，可以不做任何改动，直接使用
```

* 复习：
    * 数据类型本质是固定内存大小的别名
    * typedef，给数据类型起别名
    * 栈和堆，栈是为了效率，堆为了内存分配更加灵活
    * 栈的分配和回收由系统进行，堆的分配和回收由程序员进行
    * 数组做形参，退化为指针。数组作为形参，丢失长度信息，使用 sizeif(a)/sizeof(a[0]) 无法计算出数组长度
    * strcpy(p,"abcdefg"); 实际上不是给指针赋值，而是把字符串 "abcdefg" 拷贝到指针 p 指向的内存空间*
* 字符串，通过首地址可以用 printf 打印出来，而数组不可以，是因为字符串末尾有字符串结束符，而数组没有
* 指针变量和指针指向的内存是两个不同的概念
    * 改变指针变量的值，会改变指针的指向，但是不会改变指针指向的内存的内容
    * 改变指针指向的内存的内容，不会改变影响到指针变量的值
* 使用指针写内存的时候，一定要确保内存可写

```c
char *buf="nscnscnwsicw";	//指针直接指向文字常量区
buf2[2]='l';			//err，因为这个字符串存放在文字常量区，内容不可改

char str[]="nsijacnicwc";	//字符串常量本身存放在文字常量区，但是由于字符数组赋值，会复制一份存放在栈区
str[2]='l';				//OK，由于 str 是存放在栈区的字符数组，因此是可修改的

```

* 指针是一种数据类型，是指它指向的内存空间的数据类型

    * 指针步长 （p++），根据指针所指向的内存空间的数据类型来确定

    ```c
    p++ 等价于 (unsigned char)p+sizeof(a);
    ```

* 不允许向 NULL 和未知非法地址拷贝内存

```c
char *p3=NULL;
strcpy(p3,"lll");		//err，如果给 p3 赋值为某一个具体的非法地址，如 0x0001，也会出错，因为这个内存不允许使用
//给 p3 指向的内存区域拷贝内存，但是 p3 为空，没有指向任何有效的内存，因此内存拷贝会出错
```

#### 2.3、通过指针间接赋值

* 步骤：
    * 一般变量和指针变量
    * 建立关系
    * 通过 *  操作内存

```c
int main()
{
    int a=100;
    int *p=NULL;
    
    //建立关系，指针指向谁，就把谁的地址赋给指针了
    p=&a;
    
    //通过 *  操作内存；
    *p=32
        
    return 0;
}
```

* 如果想通过形参改变实参的值，必须地址传递

```c
int get_a()
{
    int a=10;
    return a;
}

void get_a2(int a)
{
    a=22;
}

void get_a3(int *a)
{
    *a=33;		//通过星号操作内存
}

void get_a4(int *a1,int *a2,int *a3,int *a4)
{
    *a1=33;		//通过星号操作内存
    *a2=44;
    *a3=55;
    *a4=66;
}

int main()
{

    int a=get_a();
    printf("%d\n",a);		//输出为：10

    get_a2(a);
    printf("%d\n",a);		//输出为：10
    
    //如果想通过形参改变实参的值，必须地址传递
    //实参，形参
    get_a2(a);				//在函数调用时，建立关系
    printf("%d\n",a);		//输出为：33
    
    int a1,a2,a3,a4;
    get_a4(&a1,&a2,&a3,&a4);
    printf("%d %d %d %d\n",a1,a2,a3,a4);
    
    
    
    return 0;
}
```

* 间接赋值是指针的最大意义，尤其是配合函数使用的时候
* 二级指针间接赋值

```c
void func1(int *p)
{
    p=0xaabb;
    printf("%p\n",p);		//0000aabb
}

void func2(int **p)
{
    *p=0xeeff;				//需要深入理解
    printf("%p\n",p);		//0000eeff
}

int main()
{
    /*
    //一个变量应该定义一个什么类型的指针保存它的地址
    //在原来的基础上再多加一个*
    int a=10;
    int *p=&a;
    int **q=&p;
    
    int *********t=NULL;
    int **********tp=&t;
    */
    int *p=0x1122;
    printf("%p\n",p);		//00001122
    
    func1(p);				//值传递,传递的是指针变量的值
    printf("%p\n",p);		//000011222
    
    func2(&p);				//地址传递，传递的是指针变量的地址
    printf("%p\n",p);		//0000eeff
    
    return 0;
}
```

#### 2.3、指针作为函数参数的输入输出特性

* 主调函数可以把堆区、栈区、全局数据内存地址传给被调函数
* 被调函数只能返回堆区、全局数据
* 指针作为函数参数具有输入输出特性：
    * 输入：主调函数分配内存
    * 输出：被调函数分配内存

```c
void func(char* p)
{
    //给p指向的内存区域拷贝，实际上就是main中的buf
    strcpy(p,"ssvcscac");
}

void func1(char **p,int *len)
{
    if(p==NULL)
        return;
    
    char* tmp=(char*)malloc(100);
    if(tmp==NULL)
        return;
    strcpy(tmp,"cscnsncscna");
    
    //间接赋值
    *p=tmp;
    *len=strlen(tmp);    
}

int main()
{
    //输入：主调函数分配内存
    char buf[100]={0};
    func(buf);
    printf("%s\n",buf);		//ssvcscac
    
    char *p=NULL;
    func(p);		//err，不能给空或者非法未知内存拷贝
    
    //输出：被调用函数分配内存,要想进行内存修改，必须进行地址传递
    char *p1=NULL;
    int len=0;
    func1(&p1,&len);
    if(p)
    	printf("%s	%d\n",p,len);		//cscnsncscna 11
    
    
    return 0;
}
```