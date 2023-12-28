---
layout: post
title: c_advance_1_string_pointer
date: 2023-12-28
---

## 字符串与指针

### 1、字符串的基本操作

1）字符串初始化

* C 语言没有字符串类型，通过字符数组模拟
* C 语言字符串，以字符 \0 或者 数字 0 结尾

```c
int main()
{
    //不指定长度，且没有结束符
    char buf[]={'a','b','c'};
    printf("%s\n",buf);				//输出完 abc 之后，会输出一串乱码，是因为没有字符数组没有指定长度，且末尾没有字符串结束符
    //指定长度，后面没有赋值的元素会自动补零
    char buf1[100]={'a','b','c'};
    printf("%s\n",buf1);			//输出：abc
    
    //所有元素都赋为零
    char buf3[100]={0};
    
    char buf4[2]={'a','b','c'};		//err，数组越界
    
    char buf5[50]={'1','a','b',0,'7'};		//{'1','a','b','\0','7'} 这样赋值也是一样的结果
    printf("%s\n",buf6);			//输出：1ab
    
    //常用的字符串初始化
    char buf6[]="sdnsjkcnsjcs";
    //strlen 计算字符串长度，不包括字符串结束符
    //sizeof 计算的是数组长度
    printf("strlen=%d\n,sizeof=%d\n",strlen(buf6),sizeof(buf6));		//12 13
    
    char buf7[100]="sdnsjkcnsjcs";
    printf("strlen=%d\n,sizeof=%d\n",strlen(buf7),sizeof(buf7));		//12 100
    
    return 0;
}
```

2）转义字符说明

* 当使用 \0 的时候，尽量不要在其后面根其他数字，容易使得 \0 和其他数字在一起组合成其他的转义字符，例如： \012 就是换行符 \n

3）字符串操作

```c
int mian()
{
	char buf[]="sdncscwckamcaskc";
    
    //数组方式访问字符串元素
    int i=0;
    for(i=0;i<strlen(buf);i++)
    {
        printf("%c",buf[i]);
    }
    printf("\n");
    
    //指针方式访问字符串元素
    //数组名就是数组首地址
    char *p=buf;
    for(i=0;i<strlen(buf);i++)
    {
        printf("%s",*(p+i));		//这种写法也可以：printf("%s",p[i]);
    }
    printf("\n");
    
    for(i=0;i<strlen(buf);i++)
    {
        printf("%s",*(buf+i));
    }
    printf("\n");
    
    //p和buf完全等价吗？
    //并不是，buf是一个指针常量，不能修改，而p可以修改，这是为了保证数组的内存可以回收（编译器实现就这么规定的）
    p++;		//OK
    buf++；		//err
        
    
    return 0;
}
```

4）字符串拷贝

```c
void my_strcpy(char *dst,char *src)
{
    int i=0;
    
    for(i=0;isrc[i]!=0;i++)
    {
        dst[i]=src[i]
    }
    dst[i]=0;
}

void my_strcpy1(char *dst,char *src)
{
    while(*src)
    {
        *dst=*src;
        src++;
        dst++
    }
    *dst=0;
}

void my_strcpy2(char *dst,char *src)
{
    while(*src)
    {
        *dst=*src;
        src++;
        dst++
    }
    *dst=0;
}

void my_strcpy5(char *dst,char *src)
{
    while(*dst++=*src++);		//先用了再加，写代码不能盲目追求代码简洁，而是应该让代码有足够好的性能和可读性
    *dst=0;
}

//完善字符串拷贝函数
//成功返回0，失败返回非零
//1、判断形参指针是否为空
//2、最好不要直接使用形参
int my_strcpy(char *dst,char *src)
{
    if(!dst||!src)
        return -1;
    
    //赋值变量，把形参备份
    char *to=dst;
    char *from=src;
     while(*from)
    {
        *to=*from;
        from++;
        to++
    }
    *to=0;
   return 0; 
}

int main()
{
    char src[]="ncscnsjicdjv";
    char dst[100];
    
    int ret=my_srtcpy(dst,src);
    if(ret)
    {
        printf("my_strcpy5 err: %d\n",ret);
        return ret;
    }
        
    return 0;
}
```

### 2、项目开发中常用字符串应用模型

1）while 和 do-while 模型

* 子串查找

```c
int main()
{
    char *p="ncsjcscszmnascnscuicakcnsucn";
    int cnt=0;
    /*
    do
    {
        p=strstr(p."xnasx");
        if(p!=NULL)
        {
            n++;				//累计个数
            //重新设计查找的起点
            p=p+strlen("xnasx");
        }
        else		//如果没有匹配的字符串，跳出循环
        {
            break;
        }
    }while(*p!=0);
    */
    char *p;
    while((p=strstr(p,"xnasx")!=NULL))
    {
        p=p+strlen("xnasx");
        cnt++;
        if(*p==0)
            break;
    }
    printf("%d\n",cnt);
    return 0;
}
```

2）两头堵模型

* 查找字符串非空格字符个数

    * 实际就是数组中常用的双指针思想

* 复习：

    * 指针也是一种数据类型，指针也是一种变量

    * 指针和指针指向的内存是两个不同的概念

    * 改变指针变量的值，会改变指针的指向，不影响指向内存的内容

    * 改变指针指向内存的内容，不影响指针的值，即指针的指向不变

    * 写内存时，一定要保证 内存可写

    * 只有地址传递，才能通过形参的修改，来影响实参的值

    * 不允许向 NULL 或者位置非法地址拷贝内存

    * void* 型的指针，建议使用的时候使用强制类型转换转化为目标类型的指针

    * 栈返回变量的值和变量的地址的区别

    * **可执行程序的生成过程：**

        * 预处理：宏定义展开、头文件展开、条件编译、此时不进行语法检查
        * 编译：检查语法、将预编译处理后的文件编译生成汇编文件
        * 汇编：将汇编文件生成目标文件（二进制文件）
        * 链接：将目标文件链接为可执行程序

        程序只有在运行时才加载到内存（由系统完成），但是某个变量具体分配多大内存，是在编译阶段就已经确定了。即，在比那一阶段做完处理之后，程序运行时系统才知道分配多大内存空间，因此，变量的空间在编译时就确定了，只是在运行时系统才知道具体分配多大内存，并分配相应的内存空间。

    * 变量内存的值和变量的地址

    * C 语言中没有字符串类型，使用字符数组进行模拟，数组的各种操作

    * 代码编写原则：在保证性能的前提下，尽可能保证好的阅读性

### 3、const 的使用

1）const 修饰变量

* const：修饰一个变量为只读

```c
typedef struct
{
    int a;
    int b;
}myStruct;
void func(myStruct *p)
{
    //指针能变
    //指针指向的内存的内容也可变
    p->a=10;		//OK
    
}
void func2( const myStruct* p)
{
    p=NULL;		//OK
    p->a=10;		//err
}
void func2(myStruct*   const p)
{
    p=NULL;		//err
    p->a=10;		//OK
}
void func2(const myStruct*   const p)
{
    myStruct tmp;
    p=NULL;		//err
    p->a=10;		//err
    tmp.a=p->a;		//OK
    
}
int main()
{
    //const：修饰一个变量为只读
    const int a=10;
    a=100;			//err
    
    //指针变量，指针指向的内存，2个不同的概念
    char buf[]="nsjcksic sdjhc";
    //从左往右看，跳过类型关键字，如果 const 后面是 *，说明指针指向的内存内容不可以修改；如果 const 后面是指针变量，说明指针变量的值不能修改
    const char* p=buf;		//这个和上面的 char buf[]="nsjcksic sdjhc"; 写法是一致的，都是指针变量可以修改，但是指针指向的内存的内容不可以修改
    //char const *p=buf; 这种写法和上面的是一致的
    p[1]='2';		//err
    p="ncjndscslkcm";		//OK
        
    char* const p2=buf;
    p2[1]="c";			//OK
    p2="cmsdcmsdics";		//err
    
    //p3为只读，指向不能变，指向的内容也不能变
    const char* const p3=buf;
    
    return 0;
}
```

* 引用另一个 .c 文件中的 const 变量
* const 修饰的变量在定义的时候必须初始化，否则后面就无法进行赋值和初始化了

```c
//如何引用另一个 .c 文件中的 const 变量
extern const int a;		//只能声明，不能再赋值
printf("%d\n",a);
```

2）C 语言的 const 是假的 const

* C 语言中，const 修饰的变量，无法通过变量名进行修改，但是可以通过指针修改 

```c
int main()
{
    //无法通过a来修改值，但是可以通过指针来修改
    const int a=10;
    a=100;		//err
    int *p=&b;
    *p=22;
    printf("%d %d\n",a,*p);		//22 22
    
    return 0;
}
```

### 4、二级指针

#### 4.1、二级指针作为输出，函数传参

```c
int getNUm(char* p)
{
    p=(char*)malloc(sizeof(char)*100);
    if(p==NULL)
        return-1;
    strcpy(p,"cnscnc");
    printf("p=%s\n",p);				//输出：cnscnc
    return 0;						//运行完毕，虽然堆空间没有释放，但是函数中指向堆空间的指针已经销毁，而主函数中的实参并没有指向堆空间，因此在主函数中输出的字符串仍旧为空
}
int getNUm(char** p)		//形参是指向 main 函数中的p的指针，
    //tmp 指向堆空间中的内存，并将文字常量区的字符串复制到 tmp 指向的堆空间，最后使 p 也指向这块堆空间，因此在 main 函数中可以输出这个字符串
{
    if(p==NULL)
        return-1;
    char *tmp=NULL;
    tmp=(char*)malloc(sizeof(char)*100);
    if(p==NULL)
        return-2;
    strcpy(tmp,"cnscnc");
    *p=tmp;
    printf("p=%s\n",p);				//输出：cnscnc
    return 0;						
}

int main()
{
    char *p=NULL;
    int ret=0;
    /*
    ret=getNUm(p);				//值传递，传递的是指针变量
    //值传递，形参的任何改变都不会影响到实参
    */
    
    ret=getNum2(&p);			//地址传递，传递的是指针变量的地址
    //形参修改会影响到实参
    if(ret!=0)
    {
        printf("getNum err:%d\n",ret);
        return ret;
    }
    printf("p=%s\n",p);				//输出：cnscnc
    if(p!=NULL)
    {
        free(p);
        p=NULL;
    }
    return 0;
}
```

#### 4.2、二级指针作为输入

1）指针数组

* 指针数组，指针的数组，是一个数组，每一个元素都是指针

```c
int main()
{
    //指针数组，指针的数组，是一个数组，每一个元素都是指针
    //每个元素都是相同类型的指针
    char *p[]={"11111111","22222222","333333333","44444444"};		//数组 p 有四个元素，每个元素四个字节大小
    char **q={"11111111","22222222","333333333","44444444"};		//err，因为q是一个指针，不能同时指向多个内存，q可以作为形参，但是不可以这么定义赋值
    int n=sizeof(p)/sizeof(p[0]);		// 16/4=4
    printf("%d\n",n);		// 4
    int i=0;
    for(i=0;i<4;i++)
    {
        printf("%s\n",p[i]);		//四个字符串逐一输出
    }
    
    char *q[10]={"11111111","22222222","333333333","44444444"};	
    int m=n=sizeof(q)/sizeof(q[0]);		// 40/4=4
    printf("%d\n",m);		// 10
    char *tmp;
    //选择法对p进行排序
    for(i=0;i<n-1;i++)
    {
        for(int j=i+1;j<n;j++)
        {
            if(strcmp(p[i],p[j])>0)
            {
                tmp=p[i];
                p[i]=p[j];
                p[j]=tmp;
            }
        }
    }
    
    return 0;
}
```

2）第一种内存模型：指针数组

```c
int test(int a[],int n);
//等价于：
int test(itn* a,int n);

int func(int *p[],int n);
//等价于：
int func(int **p,int n);
```

3）第二种内存模型：二维数组

* 二维数组，多个等长的一维数组，从存储角度看，依然是一维线性的

```c
void printf_array(char **a,inr n)		//二维数组作为参数，这么写不对
{
    for(int i=0;i<n;i++)
    {
        printf("%s\n",a[i]);		//err
    }
}
void printf_arry1(char a[][30],int n)	//二维数组作为参数，可以这么写
{
    for(int i=0;i<n;i++)
    {
        printf("%s\n",a[i]);		//OK
    }
}

void sort_arrry(char a[][30],int n)
{
    itn i=0,j=0;
    char tmp[30];
    for(i=0;i<n-1;i++)
    {
        for(j=i+1;j<n;j++)
        {
            if(strcmp(a[i],a[j])>0)
            {
                //交换的是内存块
                strcpy(tmp,a[i]);
                strcpy(a[i],a[j]);
                strcpy(a[j],tmp);
            }
        }
    }
}

char a[4][30]={"111111111","222222222","333333333","444444444"};
//a，首行地址；a+i，第 i 行地址
//a代表首行地址，首行首元素地址有区别，但是值是一样的，区别是地址的步长不一样
//首行地址 +1，移动30 个字节，首行首元素地址 +1，移动 1 个字节
//首行地址和首元素地址是一样的，注意 *(a+i),a[i],a+i的区别

int n=sizeof(a)/sizeof(a[0]);		//n=4

//二位是数组的第一个维度可以不写，但是必须满足一定的条件：必须在二维数组定义的时候初始化
char b[][30]={"111111111","222222222","333333333","444444444"};		//OK
char c[][30];			//err

printf_array(a,n);
```

4）第三种内存模型：动态生成二维数组，指针数组

```c
int main()
{
    char *p=NULL;
    p=(char*)malloc(100);
    strcpy(p,"nsjkcnsics");
    
    //10 个 char*，每个都是NULL
    char *p1[10]={0};
    for(int i=0;i<10;i++)
    {
        p[i]=(char*)malloc(100);
        strcpy(p[i],"cnijcns");
    }
    int a[10];
    int *q=(int*) malloc(10*sizeof(int));		//相当于是 q[10]
    
    //动态分配一个数组，数组每个元素都是 char*
    //char* ch[10]
    int n=3;
    char **buf=(char**)malloc(sizeod(char*)*n);		//相当于 char* buf[3];
    if(buf==NULL)
        return -1;
    for(i=0;i<n;i++)
    {
        buf[i]=(char*)malloc(30*sizeof(char));
        char str[30];
        strcpy(buf[i],"fcnijcnjwi");	
        //strcpy(buf[i],str);	
    }
    for(i=0;i<n;i++)
    {
        printf("%s\n",buf[i]);
    }
    
    
    //内存释放，先释放内层，再释放外层
    for(i=0;i<n;i++)
    {
        free(buf[i]);
        buf[i]=NULL;
    }
    if(buf!=NULL)
    {
        free(buf);
        buf=NULL;
    }
    
    return 0;
}
```

```c
char **getMem(int n)
{
    int i=0;
    char **buf=(char**)malloc(sizeod(char*)*n);		//相当于 char* buf[3];
    if(buf==NULL)
        return NULL;
    for(i=0;i<n;i++)
    {
        buf[i]=(char*)malloc(30*sizeof(char));
        char str[30];
        strcpy(buf[i],"fcnijcnjwi");	
        //strcpy(buf[i],str);	
    }
    for(i=0;i<n;i++)
    {
        printf("%s\n",buf[i]);
    }
    return buf;
}

void print_buf(char **buf,int n)
{
    int i=0;
    for(i=0;i<n;i++)
    {
        printf("%s\n",buf[i]);
    }
}

void free_buf(char **buf,int n)
{
    int i;
    for(i=0;i<n;i++)
    {
        free(buf[i]);
        buf[i]=NULL;
    }
    if(buf!=NULL)
    {
        free(buf);
        buf=NULL;
    }
}
int main()
{
    char **buf=NULL;
    int n=3;
    buf=getMem(n);
    if(buf==NULL)
    {
        printf("getMem err\n");
        return -1;
    }
    print_buf(buf,n);		//值传递
    free_buf(buf,n);		//值传递
    buf=NULL;
    return 0;
}
```

### 5、一维数组的使用

#### 5.1、一维数组的赋值

```c
int a[]={1,2,3,4,5,6,7,8};
//int b[];	//err,不指定长度，在定义的时候必须初始化
int c[100]={1,2,3,4};		//没有赋值的元素都为 0 

int n=0;
//sizeof，计算变量的类型所占的空间
//sizeof(a)=4*8=32	// 数组类型的大小，由元素个数、元素类型决定
n=sizeof(a)/sizeof(a[0]);		//n=8

//元素访问：a[i], *(a+i), 两种写法等价
//a+i 代表第 i 个元素的地址

```

#### 5.2、数组类型的定义

```c
int a[]={1,2,3,4,5,6,7,8};
//a 代表首元素地址，&a 代表整个数组的首地址，和首元素地址一样，但是他们的步长不一样
//a+1 跳一个元素，&(a+1) 跳整个数组长度
pritnf("%d %d\n",a,a+1);			// +4
pritnf("%d %d\n",&a,&(a+1));		// +32

//通过 typedef 定义一个数组类型
//有 typedef 是类型，没有 typedef 是变量
typedef int A[8];			//代表是一个数组类型，这里的 A 是一个数据类型，不是变量
//等价写法：typedef int (A)[8];

A b;		//等价于 int b[8]; 关于 b 的各种操作，与 int b[8] 的各种操作是一致的
//取了 typedef，b 替换到 A 的位置，就是他的含义

for(int i=0;i<8;i++)
{
    b[i]=i;
}

```

### 6、数组指针与指针数组

#### 6.1、数组指针：指向一个数组的指针

* 数组指针：指向一维数组的整个数组，而不是首元素

1）先定义数组类型，根据类型定义指针

```c
int main()
{
    int a[10]={0};
    //定义数组指针变量
    //1、先定义数组类型，根据类型定义指针
    typedef int A[10];		//[10]代表步长
    A *p=NULL;				//p 是一个数组指针变量
    p=&a;		//OK, &a 代表整个数组的首地址
    //p=a;		//不会报错，但是会有警告，这么写不准确，p是指针数组，而a 是数组首元素的地址
    //但是 a 和 &a 一样，a 最终也会被当做 &a，但是这么写不好
    pritnf("%d %d\n",p,p+1);		//输出的两个地址相差 40 ，步长为整个数组的长度 
    
    for(int i=0;i<10;i++)
    {
        //a[]
        //p=&a;		*p=*&a -> a
        (*p)[i]=i+1;
    }
    for(int i=0;i<10;i++)
    {
        printf("%d ",(*p)[i]);			//输出：1 2 3 4 5 6 7 8 9 10
    }
    printf("%d\n",sizeof(p));			// 4   
    return 0;
}
```

2）先定义数组指针类型，再定义变量

```c
int main()
{
    int a[10]={0};
    //和指针数组写法很类似，但是多了 ()
    //() 和 [] 优先级一样，从左往右
    //() 内有指针，说明他是一个指针，[]说明是一个数组，因此是一个数组指针，前面有 typedef，说明是类型 -> 数组指针类型
    typedef int (*P)[10];			
    P q;				//数组指针类型变量
    q=&a;
    for(int i=0;i<10;i++)
    {
        //a[]
        //p=&a;		*p=*&a -> a
        (*p)[i]=i+1;
    }
    for(int i=0;i<10;i++)
    {
        printf("%d ",(*p)[i]);			//输出：1 2 3 4 5 6 7 8 9 10
    }
    printf("%d\n",sizeof(p));			// 4  
    return 0;
}
```

3）直接定义数组指针变量

```c
int main()
{
    int a[10]={0};
	int (*q)[10];		//数组指针变量	
    q=&a;				//指向 a 数组
    for(int i=0;i<10;i++)
    {
        //a[]
        //p=&a;		*p=*&a -> a
        (*p)[i]=i+1;
    }
    for(int i=0;i<10;i++)
    {
        printf("%d ",(*p)[i]);			//输出：1 2 3 4 5 6 7 8 9 10
    }
    printf("%d\n",sizeof(p));			// 4  
    return 0;
}
```



#### 6.2、指针数组：数组，每个元素都是指针

```c
int main(int argc,char* argv[])		//argv，也是有个指针数组
    //argc 传参的个数，包括执行的可执行程序本身
    //argv 传入的参数字符串数组
{
    //[] 比 *  的优先级高，a 是一个指针数组
	char* a[]={"aaaaaaaa","bbbbbbbb","cccccccc"};
    
    for(int i=0;i<argc;i++)
    {
        printf("%s\n",argv[i]);		//命令行中运行：xxx.exe a b c dddd
        //输出：xxx.exe a b c dddd 共五个字符串
    }
    
    
    return 0;
}
```

#### 6.3、数组越界问题说明

```c
int main()
{
    int a[10]={0};
    a[11]=0;			//可能不会报错，只是因为刚好这个空间没有被使用，但是一旦工程项目大了，这个空间被使用了，就会报错
    return 0;
}
```

#### 6.4、数组指针与二维数组

1）二维数组

* 二维数组名 a：第 0 行首地址
* a+i：第 i 行首地址；等价于`&a[i]`
* 要想把该行首地址转换为该行的首元素地址，加 *， *(a+i)，等价于 a[i]
* 要想得到某个元素地址，加偏移量：*(a+i)+j；等价于 `a[i][i]`
* 要想得到某个元素的值，在该元素的地址前加 *， `*(*(a+i)+j)`，等价于`a[i][j]`

```c
int main()
{
    int a1[3][4]=
    {
        {1,2,3,4},
        {5,6,7,8},
        {9,10,11,12}
    };		//等价于：int a1[3][4]={1,2,3,4,5,6,7,8,9,10,11,12};
    printf("%d %d\n",a1,a1=1);		//相差 16
    //二维数组名，代表的是第 0 行的首地址（区别于第 0 行首元素地址，虽然他们的值一样，但是他们的步长不一样）
    printf("%d %d\n",*(a1+0)，*(a1+1));		//第 0 行首元素地址,第 1 行首元素地址，相差 16	等价于：printf("%d %d\n",a1[0],a1[1]);
    printf("%d\n",*(a1+0)+1);		//第 0 行 第一个元素地址		等价于：printf("%d\n",a[0]+1);
    //a，代表第 0 行首地址
    //a+i，代表第 i 行首地址
    //*(a+i)，等价于 a[i]，代表第 i 行首元素地址
    //*(a+i)+j，等价于 &a[i][j]，代表第 i 行第 j 列元素的地址
    //*(*(a+i)+j)，等价于 a[i][j]，代表第 i 行第 j 列的元素

    return 0;
}
```

2）二维数组（多维数组）的存储结构

* 不存在多维数据，线性存储的（可以通过打印上一行最后一个元素的地址和下一行第一个元素的地址，会发现他们是相邻存储的）
* 二维数组元素个数：`n=sizeof(a)/sizeof(a[0][0]])`
* 二维数组的行数：`sizeof(a)/sizeof(a[0])`
* 二维数组的列数：`sizeof(a[0])/sizeof(a[0][0])`

```c
int main()
{
    int a1[3][4]={1,2,3,4,5,6,7,8,9,10,11,12};
    int n=sizeof(a)/sizeof(a[0][0]]);
    for(int i=0;i<n;i++)
    {
        printf("%d ",a[i]);		//输出结果：1 2 3 4 5 6 7 8 9 10 11 12
    }
    return 0;
}
```

3）数组指针和二维数组

```c
int main()
{
    //3 个 a[4] 的一维数组
    int a[3][4]={1,2,3,4,5,6,7,8,9,10,11,12};
    
    //定义数组指针
    int (*p)[4];
    p=a;		//指向首行首地址，步长为二维数组的一行的长度，p 的等价于数组名
    //p=&a;		//指向整个二维数组的首地址，步长为整个数组的长度
    printf("%d %d\n",p,p+1);		//相差 16，步长为二维数组的一行的长度
    for(int i=0;i<3;i++)
    {
        for(int j=0;j<4;j++)
        {
            printf("%d ",*(*(p+i)+j));		//等价于：printf("%d ",p[i][j]);
        }
        printf("\n");
    }
    
    return 0;
}
```

4）首行首元素地址和首行首地址的区别

* 计算一维数组的长度：sizeof(首行首元素地址)

```c
int t[]={1,2,3,4,5,6,7,8,9,10};
primtf("%d %d\n",sizeof(t),sizeof(&t));		// 40 4	，分别是首行首元素地址和首行首地址

int a[2][10];
pritnf("%d %d\n",sizeof(a[0]),sizeof(&a[0]));		// 40 4	，分别是首行首元素地址和首行首地址
```

5）二维数组做形参

```c
void printArray(int a[3][4])
{
    for(int i=0;i<3;i++)
    {
        for(int j=0;j<4;j++)
        {
            printf("%d ",a[i][j]);
        }
    }
}

//数组指针
typedef int(*P)[]
void printArray1(int a[][4])
{
    for(int i=0;i<3;i++)
    {
        for(int j=0;j<4;j++)
        {
            printf("%d ",a[i][j]);
        }
    }
}

//数组做形参，会退化为指针
void printArray2(int (*a)[4])
{
    for(int i=0;i<3;i++)
    {
        for(int j=0;j<4;j++)
        {
            printf("%d ",a[i][j]);
        }
    }
}


int main()
{
    int a[3][4]={1,2,3,4,5,6,7,8,9,10,11,12};
    printArray(a);
    printArray1(a);
    printArray2(a);
    
    return 0;
}
```
