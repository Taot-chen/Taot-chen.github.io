---
layout: post
title: c_advance_4_linked_list_function_pointer.md
date: 2023-12-28
---

## 链表和函数指针

### 1、链表相关概念

#### 1.1、链表和数组的区别

* 链表是一种常用的数据结构，通过指针将一系列数据结点，连接成一个数据链
* 相对于数组，链表有更好的动态性（数组顺序存储，链表非顺序存储）
* 数据域用于存储数据，指针域用来建立与下一个结点的联系
* 数组一次性分配一块连续的存储区域
    * 优点：随机访问效率高
    * 缺点：
        * 如果需要分配的区域非常大，可能会分配失败
        * 删除和插入某个元素的效率低
* 链表：
    * 优点：
        * 不需要一块连续的存储区域
        * 删除和插入某个元素效率高
    * 缺点：
        * 随机访问效率低

#### 1.2、链表概念和分类

* 数据域、指针域
* 动态链表、静态链表
    * 动态链表：动态分配内存（常用）
* 带头链表、不带头链表
    * 带头链表：头结点不存储数据，只是标志位，用来指向第一个数据结点。头结点固定，第二个是有效结点
    * 不带头链表：头结点不固定
* 单向链表、双向链表、循环链表

#### 1.3、结构体套结构体

* 结构体可以嵌套另外一个结构体的任意类型变量
* 结构体不可以嵌套本结构体的普通变量（因为此时本结构体还未定义完成，大小还不确定，而类型的本质是固定大小内存块别名）
* 结构体可以嵌套本结构体的指针变量（因为指针变量的空间是确定的，此时结构体的大小是可以确定的）

```c
typedef struct A
{
    int a;
    int b;
    char *p;
}A;
//结构体可以嵌套另外一个结构体的任意类型变量
//结构体不可以嵌套本结构体的普通变量（因为此时本结构体还未定义完成，大小还不确定，而类型的本质是固定大小内存块别名）
//结构体可以嵌套本结构体的指针变量（因为指针变量的空间是确定的，此时结构体的大小是可以确定的）
typedef struct B
{
    int a;
    A tmp1;		//OK
    A *p1;		//OK
    B b;		//err
}B;
```

### 2、静态链表的使用

```c
typedef struct Stu
{
    int id;		//数据域
    char name[100];
    struct Stu *next;	//指针域
}Stu;

int main()
{
    //初始化三个结构体变量
    Stu s1={1,"mkle",NULL};
    Stu s2={2,"lily",NULL};
    Stu s2={3,"lilei",NULL};
    
    s1.next=&s2;
    s2.next=&s3;
    s3.next=NULL;		//尾结点
    
    Stu *p=&s1;
    while(p!=NULL)
    {
        printf("%d %s\n",p->id,p->name);
        
        // p 移动到下一个结点
        p=p->next;
    }
    
    return 0;
}
```

### 3、动态链表

#### 3.1、单向链表的基本操作

```c
typedef struct Node
{
    int id;
    struct Node *next;
}Node;

//创建头结点,链表创建
//链表的头结点地址由函数返回
Node *SListCreat()
{
    Node *head=NULL;
    //头结点作为标志，不存储数据
    head=(Node*)malloc(sizeof(Node));
    if(head==NULL)
        return NULL;
    //给head的成员变量赋值
    head->id=-1;
    head->next=NULL;
    
    Node *pCur=head;
    Node *pNew=NULL;
    
    int data;
    while(1)
    {
        printf("请输入数据：");
        scanf("%d",&data);
        if(data==-1)		//输入-1，退出输入
            break;
        //新节点动态分配空间
        pNew=(Node*)malloc(sizeof(Node));
        if(pNew==NULL)
            continue;
        //给pNew成员变量赋值
        pNew->id=data;
        pNew->next=NULL;
        
        //链表建立关系
        //当前节点的next指向pNew
        pCur->next=pNew;
        //pNew下一个结点为空
        pNew->next=NULL;
        
        //把pCur移动到pNew
        pCur=pNew;
        
    }
    return head;
}


//链表的遍历
int SListPrint(Node* head)
{
    if(head==NULL)
        return -1;
    //取出第一个有效结点，即头结点的next
    Node *pCur=head->next;
    
    while(pCur!=NULL)
    {
        printf("%d->",pCur->id);
        
        //当前节点移动到下一个结点
        pCur=pCur->next;
    }
    printf("NULL\n");
    return 0;
}


//链表插入结点
//在值为 x 的结点前，插入值为 y 的结点，若值为 x 的结点不存在，则插在链表尾部
int SListNodeInsert(Node* head,int x,int y)
{
    if(head==NULL)
        return -1;
    Node *pPre=head;
    NOde *pCur=head->next;
    
    while(pCur!=NULL)
    {
        if(pCur->id==x)
            break;
        
        //不相等，两个指针同时后移一个结点
        pPre=pCur;
        pCur=pCur->next;
    }
    //给新结点动态分配空间
    Node *pNew=(Node*)malloc(sizeof(Node));
    if(pNew==NULL)
        return -2;
    
    //给pNew的成员变量赋值
    pNew->id=y;
    pNew->next=NULL;
    
    //插入到指定位置
    pPre->next=pNew;
    pNew->next=pCur;
    
    return 0;
}



//删除指定的结点
//删除第一个值为x的结点
int SListNodeDelete(Node *head,int x)
{
    if(head==NULL)
        return -1;
    Node *pPre=head;
    NOde *pCur=head->next;
    int flag=0;		//0表示没有找到，1 表示找到了
    
    while(pCur!=NULL)
    {
        if(pCur->id==x)
        {
            //删除结点操作
            pPre->next=pCur->next;		
            free(pCur);
            pCur=NULL;
            flag=1;
            break;
        }

        //不相等，两个指针同时后移一个结点
        pPre=pCur;
        pCur=pCur->next;
    }
    if(flag==0)
    {
        printf("没有 id 为 %d 的结点\n",x);
        return -2;
    }
    return 0;    
}



//清空链表，释放所有结点
int SlistNodeDestroy(Node *head)
{
    if(head==NULL)
        return -1;
    Node *tmp=NULL;
    while(head!=NULL)
    {
        tmp=head->next;
        free(head);
        head=NULL;
        
        head=tmp;
    }
    return 0;
}

int main()
{
	Node *head=NULL;
    head=SListCreat();		//创建头结点
    int res=SListPrint(head);
    
    //在5的前面插入4
    SListNodeInsert(head,5,4);
    SListPrint(head);
    
    SListNodeDelete(head,5);		//删除第一个值为x的结点
    SListPrint(head);
    
    //链表清空
    SlistNodeDestroy(head);
    head=NULL;
    
    return  0;
}
```

### 4、函数指针

#### 4.1、指针函数：返回指针类型的函数

```c
//指针函数
//() 的优先级比 * 高，因此 int* func() 是一个函数，返回值是 int 型的指针
int* func()
{
    int *p=(int*)malloc(sizeof(int));
    return p;
}

int main()
{
    int *p=func();
    return 0;
}
```

#### 4.2、函数指针：是指针，指向函数的指针

1）函数指针的定义方式

* 定义函数类型，根据类型定义指针变量（有 typedef 是类型，没有是变量）
* 先定义函数指针类型，根据类型定义指针变量（常用）
* 直接定义函数指针变量（常用）

```c
int func(int a)
{
    printf("a=========%d\n",a);
    return 0;
}

int main()
{
    
    //1、定义函数类型，根据类型定义指针变量（有 typedef 是类型，没有是变量），这种方式不常用
    typedef int FUNC(int a);		//FUNC 就是函数类型
    FUNC *p1=NULL;					//函数指针变量，且有要求，要求 p1 指向的函数返回值是 int 类型，且只有一个参数，参数类型是 int
    p1=func;		//等价于：p1=&func;		p1 指向 func 函数
    func(5);		//传统的函数调用
    p1(6);			//函数指针变量调用方式
    
    
    
    //2、先定义函数指针类型，根据类型定义指针变量（常用）
    typedef int (*FUNC)(int a);			//FUNC，就是函数指针类型
    FUNC p2=func;		//p2 指向函数 func
    p2(7);
    
    
    
    //3、直接定义函数指针变量（常用）
    int (*p3)(int a)=func;		//p3 指向函数 func
    p3(8);
    
    int (*p4)(int a);
    p4=func;					//p4 指向函数 func
    p4(9);
    
    
    return 0;
}
```

2）函数指针的应用

```c
int add(int x,int y)
{
    return x+y;
}

int sub(int x,int y)
{
    return x-y;
}

int multi(int x,int y)
{
    return x*y;
}

int divide(int x,int y)
{
    if(y==0)
    {
        printf("除数不可以是0\n");
        return 0;
    }
    return x/y;
}

void myExit()
{
    exit(0);
}

int main()
{
    char cmd[100];
    while(1)
    {
        printf("请输入指令：");
        scanf("%s",cmd);
        if(!strcmp(cmd,"add"))
            add(1,2);
        else if(!strcmp(cmd,"sub"))
            sub(1,2);
        else if(!strcmp(cmd,"multi"))
            multi(1,2);
        else if(!strcmp(cmd,"divide"))
            divide(1,2);
        else if(!strcmp(cmd,"myExit"))
            myExit();
        else 
            pritnf("Wrong input!\n")
    }
    
    
    //使用函数指针数组调用函数
    int (*func[4])(int a,int b)={add,sub,multi,divide};
    char *buf[]={"add","sub","multi","divide"};
    while(1)
    {
        printf("请输入指令：");
        scanf("%s",cmd);
        if(!strcmp(cmd,"myExit"))
            myExit();
        else 
            pritnf("Wrong input!\n")
        for(int i=0;i<4;i++)
        {
            if(strcmp(cmd,buf[i])==0)
            {
                func[i](1,2);
                break;				// 跳出for
            }
                
        }
    }
    
    return 0;
}
```

3）回调函数的使用

```c
int add(int x,int y)
{
    return x+y;
}

//在17:11，添加减法功能，则可以直接使用之前的框架，调用减法功能
int sub(int x,int y)
{
    return x-y;
}


//函数的参数是变量，可以使函数指针变量吗？
//框架，固定不变，完成coding时间为 17：10
//C++ 的多态便是如此，调用相同的接口，执行不同的功能
void func(int x,int y,int(*p)(int a,int b))
{
    printf("func111111111\n");
    int res=p(x,y);		//回调函数
    printf("%d\n",a);
}


//上面的等价写法：
typedef int(*Q)(int a,int b);		//函数指针类型
void func2(int x,int y,Q p)
{
    printf("func111111111\n");
    int res=p(x,y);		//回调函数
    printf("%d\n",a);
}



int main()
{
    func(1,2,add);		//输出结果：func111111111\n 3		//1+2=3
    func(1,2,sub);		//输出结果：func111111111\n -1		//1-2=-1
    return 0;
}
```

#### 4.3、链表的内存四区

* 链表结点交换有两种方法：
    * 第一种方法：分别交换两个链表的数据域和指针域
    * 第二种方法：只交换两个链表的数据，将数据封装为结构体，这样能够将数据封装为一个，简化交换的过程

#### 4.4、删除指定的所有结点

```c
//删除值为 x 的所有结点
int SListNodeDeletePro(Node* head,int x)
{
    if(head==NULL)
        return -1;
    Node *pPre=head;
    NOde *pCur=head->next;
    int flag=0;		//0表示没有找到，1 表示找到了
    
    while(pCur!=NULL)
    {
        if(pCur->id==x)
        {
            //删除结点操作
            pPre->next=pCur->next;		
            free(pCur);
            pCur=NULL;
            flag=1;
            pCur=pPre->next;
            continue;		//如果相等，删除结点，并特跳出本次循环，避免后续再次两指针同时后移一个结点
        }

        //不相等，两个指针同时后移一个结点
        pPre=pCur;
        pCur=pCur->next;
    }
    if(flag==0)
    {
        printf("没有 id 为 %d 的结点\n",x);
        return -2;
    }
    return 0;   
}
```



#### 4.5、链表排序

```c
int SListNodeSort(Node* head)
{
    if(head==NULL||head->next==NULL)
    {
        return -1;
    }
    Node *pPre=NULL;
    Node *pCur=NULL;
    Node tmp;
    for(pPre=head->next;pPre->next!=NULL;pPer=pPre->nxt)
    {
        for(pCur=pPre->next;pCur!=NUll;pCur=pCur->next)
        {
            if(pPre->id>pCur->id)
            {
                /*
                //交换数据域
                tmp=*pPre;
                *pPre=*pCur;
                *pCur=tmp;
                
                //交换指针域
                tmp.next=pCur->next;
                pCur->next=pPre->next;
                pPre->next=tmp.next;
                */
                
                //节点只有一个数据，可以只交换数据域，比较容易实现
                tmp.id=pCur->id;
                pCur->id=pPre->id;
                pPre->id=tmp.id;
            }
        }
    }
    return 0;
}
```



#### 4.6、升序插入链表结点

```c
int SListNodeInsertPro(Node *head,int x)
{
    //保证插入前就是有序的
    itn retr=SListNodeSort(head);
    if(ret!=0)
        return ret;
    
    //插入结点
    Node *pPre=head;
    NOde *pCur=head->next;
    
    //1 2 3 5 6，插入4
    //pre：3,	cur：5
    while(pCur!=NULL)
    {
        if(pCur->id>x)		//找到了插入点
            break;
        
        //不相等，两个指针同时后移一个结点
        pPre=pCur;
        pCur=pCur->next;
    }
    //给新结点动态分配空间
    Node *pNew=(Node*)malloc(sizeof(Node));
    if(pNew==NULL)
        return -2;
    
    //给pNew的成员变量赋值
    pNew->id=y;
    pNew->next=NULL;
    
    //插入到指定位置
    pPre->next=pNew;
    pNew->next=pCur;
    
    return 0;
}
```



#### 4.7、链表的翻转

```c
int SListNodeReverse(Node* head)
{
    if(head==NULL||head->next==NULL||head->next->next==NULL)
        return -1;
    Node *pPer=head->next;
    Node *pCur=pPre->next;
    Node *tmp=NULL;
    
    while(pCur!=NULL)
    {
        tmp=pCur->next;
        pCur->next=pPre;
        
        pPre=pCur;
        pCur=tmp;        
    }
    
    //头结点和之前的第一个节点的指针域的处理
    head->next->next=NULL;
    head->next=pPre;
    
    return 0;
}
```



### 5、预处理

#### 5.1、预处理

* C 语言 对源程序处理的四个步骤：预处理、编译、汇编、链接
* 预处理是在程序源代码被编译之前，由预处理器对程序源代码进行的处理。这个过程不对程序的源代码进行语法解析，但会把源代码分割或处理为特定的符号为下一步的编译做准备。

#### 5.2、预编译命令

* C 编译器提供的预处理功能主要有四种：
    * 文件包含：#include
    * 宏定义：#define
    * 条件编译：#if #endif
    * 一些特殊作用的预定义宏
* #include<> 和 #include"" 的区别
    * <> 表示系统直接按照系统指定的目录检索
    * "" 表示系统现在 file.c 所在的当前目录找 file.h，如果找不到，再按照系统指定的目录检索
    * 注意：
        * #include<> 常用于包含库函数的头文件
        * #include"" 常用于包含自定义的头文件
        * 理论上 #include 可以包含任何格式的文件（.c .h 等），但一般用于头文件的包含

#### 5.3、宏定义

* 在源程序中，允许一个标识符（宏名）来表示一个语言符号字符串，用指定的符号代替指定的信息

* 在 C 语言中，宏分为无参数的宏和有参数的宏

* 宏的作用域：宏可以写在程序的任何地方，但是都是类似于是全局的。只要定义了宏，宏定义后面的代码都可以使用

* 取消宏定义：在定义过宏之后，可以取消宏定义，取消之后面的代码都不能使用这个宏了

    ```c
    //宏定义
    #define PI 3.14
    int r=10;
    double area=PI*r*r;
    
    //取消宏定义
    #undef PI 
    
    ```

    

1）无参数的宏

```c
#define 宏名 字符串
```

```c
#include<stdio.h>

#define PI 3.14

int main()
{
    int r=10;
    double area=PI*r*r;
    
    return 0;
}
```

2）有参数的宏

```c
#define TEST(a,b) a*b
int main()
{
    int a=TEST(1,2);		//相当于： a=1*2;
    //但是宏只进行简单的替换，因此，这个宏最好这么写：#define TEST(a,b) (a)*(b)
        
    return 0;
}

```

3）宏定义函数

```c
//宏定义比较两个数的大小,返回较大的数
#define MAX2(a,b) (a)>(b)?(a):(b)

//宏定义比较三个数的大小,返回最大的数
#define MAX3(a,b,c) (a)>(MAX2(b,c))?(a):(MAX2(b,c))		//#define MAX3(a,b,c) (a)>MAX2(b,c)?(a):MAX2(b,c) 这样写是不对的，宏展开的时候直接替换，MAX2(b,c) 也是直接替换，结果不对
```



#### 5.4、条件编译

1）条件编译

* 一般情况下，源程序中所有的行都参加编译，但是有时候希望部分程序行只在满足一定条件时才编译，即对这部分源程序航指定编译条件：

    * 测试存在：一般用于调试，裁剪

    ```c
    # ifdef 标识符
    	程序段1
    # else
        程序段2
    # endif    
    ```

    ```c
    #define D1
    
    #ifdef D1
    	printf("D111111111111\n");
    #else
    	printf("others\n");
    #endif
    ```

    

    * 测试不存在：一般用于头文件，防止重复包含

    ```c
    # ifndef 标识符
    	程序段1
    # else
        程序段2
    # endif
    ```

    ```c
    //#pragma once				//比较新的编译器支持，老的编译器不支持
    
    
    //老的编译器支持的写法
    //__SOMEHEAD_H__ 是自定义宏，每个头文件的宏都不一样
    //一般都这么写：
    //test.h -> _TEST_H_
    //fun.h -> _FUN_H_
    #ifndef __SOMEHEAD_H__
    #define __SOMEHEAD_H__
    
    //函数声明
    
    
    #endif      		//！__SOMEHEAD_H__
    ```

    

    * 根据表达式定义

    ```c
    # if 表达式
    	程序段1
    # else
    	程序段2
    # endif
    ```

    ```c
    #define TEST 1
    
    #if TEST
    	printf("1111111111111\n");
    #else
    	pritnf("22222222222222\n");
    #endif
    ```

    

### 6、递归

* 递归：函数可以调用函数本身（不要用 main 函数调用 main 函数，不是不可以，只是这样做往往得不到想要的结果）

* 普通函数调用：栈结构，先进后出，先调用后结束

* 函数递归调用：调用流程与普通函数的调用是一致的

* 递归函数一定要注意递归结束条件的设置。

    ```c
     int add(int n)
     {
         if(n==100)
             return n;
         return n+add(n+1);
     }
    
    int main()
    {
        int n=100;
        int sum=0;
        sum=add(1);
        
        return 0;
    }
    ```

1）递归实现字符串翻转

```c
int reverseStr(char *str)		//递归的结果放在全局变量内
{
    if(str==NULL)
        return -1;
    
    if(*str=='\0')		//递归结束条件
        return 0;
    
    if(inverseStr(str+1)<0)
        return -1;
    strcat(g_buf,str,1);
    
    return 0;
}

int main()
{
	    
    return 0;
}
```



### 7、函数动态库封装

* 封装的时候，不需要主函数，只需要实现的功能的 `.c` 文件和 `.h` 文件即可

* 动态库不可以有中文路径

* 在 Windows 下，VS 里面，新建项目（路径不能有中文）：testdll（win32 控制台应用程序）（放在 文件夹 testdll 中） $\rightarrow$ 下一步 $\rightarrow$ DLL、空项目 $\rightarrow$ 完成

    * 要生成动态库的代码在 socketclient.c 和 socketclient.h 中，不需要主函数，将这两个文件放在前面新建的 testdll 所在的文件夹 testdll 中，在 testdll 文件夹中会自动生成一个 testdll 子文件夹，将  socketclient.c 和 socketclient.h 放在这个子文件夹中
    * 添加现有项，选中 socketclient.c 和 socketclient.h 并添加进来
    * 对于  socketclient.c 和 socketclient.h 中的每一个函数，需要在函数定义钱脉添加：__declspet(dllexport) 。如果某个函数前面没有放，生成的动态库里面就不包含该函数，调用该函数会失败
    * 编译，会提示没有主函数，可忽略该提示即可
    * 找到前面的 testdll 子文件夹，进入其中的 Debug 文件夹，里面有两个很重要的文件：testdll.dll 和 testdll.lib。前者是程序运行的时候使用，后者是编译程序的时候使用
    * 使用的时候，单独拷贝出 testdll.dll 和 testdll.lib，以及 socketclient.h， socketclient.h，的作用是为用户提供一个函数的声明说明
    * 在其他项目中，需要使用该动态库的时候，将 testdll.lib，以及 socketclient.h 拷贝到项目所在文件夹，并把头文件添加到 **头文件** 文件夹中 $\rightarrow$ 在 VS 中右键项目文件夹（此时无法编译通过） $\rightarrow$ 属性（或者：项目 $\rightarrow$ 属性）$\rightarrow$ 配置属性 $\rightarrow$ 链接器 $\rightarrow$ 输入 $\rightarrow$ 附加依赖项 $\rightarrow$ testdll.lib $\rightarrow$ 确定 $\rightarrow$ 应用（此时可以编译通过了，但是无法运行） $\rightarrow$ testdll.dll 拷贝到项目所在文件夹 $\rightarrow$ 此时就可以编译运行通过了

* 对于安装的程序，在安装路径中，会有很多的 .dll 文件，也都是动态库

    ```c
    // 加法
    __declspet(dllexport) 			//表示这个函数导出为动态库
    int addAB(int a,int b)
    {
        return a+b;
    }
    ```



### 8、内存泄漏检测

#### 8.1、日志打印

* C 语言中的一些常用的与日志打印相关的预定义的宏

```c
#define _CRT_SECURE_NO_WARNINGS

// __FILE__：打印这个语句所在的文件的绝对路径
// __LINE__：打印这个语句所在的行号
printf("%s, %d\n",__FILE__,__LINE__);
```

* 实际开发中，会有专门的日志打印相关的工具

#### 8.2、内存泄漏检查

* 实际开发中会有专门的内存泄漏检测相关的工具
* 有很多好用的开源框架可以使用