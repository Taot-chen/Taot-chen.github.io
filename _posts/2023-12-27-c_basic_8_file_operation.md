---
layout: post
title: c_basic_8_file_operation
date: 2023-12-27
---

## 文件操作

### 1、概述

#### 1.1、磁盘文件和设备文件

1）磁盘文件：

指一组相关数据的有序集合，通常存储在外部介质（如磁盘）上，使用时才调入内存

2）设备文件：

在操作系统中，把每一个与主机相连的输入、输出设备看作是一个文件，把它们的输入、输出等同于文件的读写

#### 1.2、磁盘文件的分类

* 物理上所有的磁盘文件本质上都是一样的：以字节为单位进行顺序存储
* 从用户或者操作系统使用的角度（逻辑上）把文件分为：
    * 文本文件：基于字符编码的文件
    * 二进制文件：基于值编码的文件

#### 1.3、文本文件和二进制文件

1）文本文件

* 基于字符编码，常见编码有 ASCII、UNICODE等
* 一般可以使用文本编辑器直接打开
* 数 5678 以 ASCII 存储形式为：00110101 00110110 00110111 00111000

2）二进制文件

* 基于值编码，自己根据具体应用，指定某个值是什么意思
* 把内存中的数据按其在内存中的存储形式鸳鸯输出到磁盘上
* 数 5678 的存储形式（二进制码）为：00010110 00101110

### 2、文件的打开和关闭

#### 2.1、文件指针

* 在 C 语言中，用一个指针变量指向一个文件，这个指针称为文件指针
* FILE 是系统使用 typedef 定义出来的有关文件信息的一种结构体类型，结构体中含有文件名、文件状态和文件当前位置等信息
* 声明 FILE 结构体的信息包含在 stdio.h 头文件中
* C 语言中有三个特殊的文件指针由系统默认打开，用户无需定义即可直接使用：
    * stdin：标准输入，默认为当前终端（键盘），使用 scanf、getchar 函数默认从此终端获得数据
    * stdout：标准输出，默认为当前终端（屏幕），使用 printf、puts 函数默认输出信息到此终端
    * stderr：标准出错，默认为当前终端（屏幕），使用 perror 函数默认输出信息到此终端

```c
//FILE 结构体内容
typedef struct
{
    short level;			//缓冲区满或者空的程度
    unsigned flags;			//文件状态标志
    char fd;				//文件描述符
    unsigned char hold;		//如无缓冲区，不读取字符
    short bsize;			//缓冲区的大小
    unsigned char *buffer;	//数据缓冲区的位置
    unsigned ar;			//指针当前的指向
    unsigned istemp;		//临时文件指示器
    short token;			//用于有效的检查
}FILE;
```

#### 2.2、文件打开

任何文件在使用之前都应该先打开：

```c
#include<stdio.h>
FILE* fopen(const char* filename,const char* mode);
```

* 功能：打开文件
* 参数：
    * filename：需要打开的文件名，根据需要加上路径
    * mode：打开文件的模式设置
* 返回值：
    * 成功：文件指针
    * 失败：NULL

3）文件打开模式的几种形式：

|  打开模式   |                             含义                             |
| :---------: | :----------------------------------------------------------: |
|  r 或者 rb  | 以只读方式打开一个文本文件（不创建文件，若文件不存在则报错） |
|  w 或者 wb  | 以写方式打开文件（如果文件存在则清空文件，如果文件不存在则创建一个文件） |
|  a 或者 ab  |  以追加方式打开文件，在末尾添加内容，若文件不存在则创建文件  |
| r+ 或者 rb+ |          以可读、可写的方式打开文件（不创建新文件）          |
| w+ 或者 wb+ | 以可读、可写的方式打开文件（如果文件存在则清空文件，如果文件不存在则创建一个文件） |
| a+ 或者 ab+ | 以添加方式打开文件，并在文件末尾更改文件，若文件不存在则创建文件 |

**注意：** 

* b 是二进制的意思，b 只在 windows 下有效，在 Linux 用 r 和 rb 的结果是一样的
* Unix 和 Linux 下所有的文本文件行都是以 \n 结尾，而 Windows 所有的文本文件都是以 \r\n 结尾
* 在 Windows 平台下，以”文本“方式打开文件，不加 b：
    * 当读取文件的时候，系统会将所有的 \r\n 转换成 \n
    * 当文件写入的时候，系统会将 \n 转换成 \r\n 写入
    * 以二进制方式打开文件，则读写过程都不会进行 \n 和 \r\n 的转换
* 在 Unix 和 Linux 平台下，“文本”与“二进制”模式没有区别，\r\n 作为两个字符原样输入输出

```c
int main()
{
    FILE* fp=fopen("D:/a.txt","r");
    if(fp==NULL)
    {
        printf("打开文件失败\n");
        //原因：
        //1、找不到文件
        //2、文件的权限不满足（文件权限：r, w, x）
        //3、程序打开状态的文件数超出上限（65535）
        return -1;
    }
    printf("打开文件成功：%p\n",fp);	//输出的是一个地址
    
    fclose(fp);		//在使用文件指针的过程中，关闭当前文件之前，不要去改变文件指针的指向，因为改变之后，打开的文件就无法关闭了
    return 0;
}
```

#### 2.3、文件关闭

任何文件在使用后应该关闭：

* 打开的文件会占用内存资源，如果总是打开不关闭，会消耗很多内存
* 一个进程同时打开的文件数是有限的，超过最大同时打开文件数，再次调用 fopen 打开文件会失败
* 如果没有明确调用 fclose 关闭打开的文件，那么程序在退出的时候，操作系统会同一关闭

```c
#include<stdio.h>
int fclose(FILE* stream);
```

* 功能：关闭先前 fopen 打开的文件，此动作让缓冲区的数据写入文件中，并释放系统提供的文件资源
* 参数：stream，文件指针
* 返回值：
    * 成功：0
    * 失败：-1

### 3、文件的顺序读写

#### 3.1、按照字符读写文件 fgetc、fputc

1）写文件

```c
#include<stdio.h>
int fputc(int ch,FILE* stream);

```

* 功能：将 ch 转换为 unsigned char 后写入 stream 指定的文件中
* 参数：
    * ch：需要写入文件的字符
    * stream：文件指针
* 返回值：
    * 成功：成功写入文件的字符
    * 失败：返回-1

**在写入的过程中，光标在写入一个字符完成后，光标会自动向后移动一位**

```c
int main()
{
    //以写的方式打开文件，如果文件存在则清空文件，如果文件不存在则创建一个文件
    FILE* fp=fopen("D:/a.txt","w");
    if(!fp)
    {
        printf("文件打开失败\n");
        return -1;
    }
    char ch='a';
    //字符写入
    fputc(ch,fp);
    
    
	fclose(fp);    
    return 0;
}
```

```c
int main()
{
    //以写的方式打开文件，如果文件存在则清空文件，如果文件不存在则创建一个文件
    FILE* fp=fopen("D:/a.txt","w");
    if(!fp)
    {
        printf("文件打开失败\n");
        return -1;
    }
    
    char ch;
    while(1)
    {
        scanf("%c",&ch);
        if(ch=='@')
            break;
        fputc(ch,fp);
    }
    
    fclose(fp);
    return 0;
}
```



2）读文件

```c
#include<stdio.h>
int fgetc(FILE* stream);
```

* 功能：从 stream 指定的文件中读取一个字符
* 参数：
    * stream：文件指针
* 返回值：
    * 成功：返回读取到的字符
    * 失败：返回-1

* * 

```c
//文件字符读写
int main()
{
     FILE* fp=fopen("D:/a.txt","r");
    if(!fp)
    {
        printf("文件打开失败\n");
        return -1;
    }
    
    char ch;
    //文件的字符读取
    ch=fgetc(fp);
    printf("%c\n",ch);
    //读取下一个，不需要改变文件指针，每次读取完一个字符之后，光标会自动移动到下一个字符处
    //这里不要手动改变文件指针的值，否则后面文件关闭难以实现
    //文件默认结尾为 -1，即文件以 -1 作为结尾标志
    ch=fgetc(fp);
    printf("%c\n",ch);
    
    //关闭文件
    fclose(fp);
    
    return 0;
}
```

```c
//文件字符读写
int main()
{
     FILE* fp=fopen("D:/a.txt","r");
    if(!fp)
    {
        printf("文件打开失败\n");
        return -1;
    }
    
    char ch;
    while((ch=fgetc(fp))!=EOF)
    {
        printf("%c",ch);
    }
    
    //关闭文件
    fclose(fp);
    
    return 0;
}
```



3）文件结尾

* 在 C 语言中，EOF 表示文件结束符，在 while 循环中以 EOF 作为文件结束标志，EOF 是一个宏，其值为 -1
* 这种以 EOF 作为文件结束标志的文件，必须是文本文件
* 当把数据以二进制形式存放到文件中时，就会有 -1 出现，此时不能使用 EOF 作为二进制文件的结束标志。可以使用 feof  函数来判断文件是否结束
* feof 函数既可以判断二进制文件，又可以判断文本文件

```c
#define EOF (-1)
```

```c
#include<stdio.h>
int feof(FILE* stream);

```

* 功能：检测是否读取到了文件结尾。判断的是最后一次 **读取操作的内容**，不是当前位置内容（上一个读取内容）
* 参数：stream，文件指针
* 返回值：
    * 非 0 值，已经到文件结尾
    * 0，没有到文件结尾

#### 3.2、文件加密解密

```c
//加密
int main()
{
    FILE* fp1=fopen("D:/解密.txt","r");
    FILE* fp2=fopen("D:/加密.txt","w");
    if(!fp1||!fp2)
    {
        return -1;
    }
    
    char ch;
    while((ch=fgetc(fp1))!=EOF)
    {
        ch++;
        fputc(ch,fp2);
    }
    fclose(fp1);
    fclose(fp2);
    
    return 0;
}


```

```c
//解密
int main()
{
    FILE* fp1=fopen("D:/加密.txt","r");
    FILE* fp2=fopen("D:/解密文件.txt","w");
    if(!fp1||!fp2)
    {
        return -1;
    }
    
    char ch;
    while((ch=fgetc(fp1))!=EOF)
    {
        ch--;
        fputc(ch,fp2);
    }
    fclose(fp1);
    fclose(fp2);
    
    return 0;
}
```

#### 3.3、文件行读写 fgets、fputs

1）写文件：fputs

```c
#include<stdio.h>
int fputs(const char *str,FILE *stream);
```

* 功能：将 str 所指定的字符串写入到 stream 指定的文件中，字符串结束符 '\0' 不写入文件
* 参数：
    * str，字符串
    * stream，文件指针，如果把字符串输出到屏幕，固定写为 stdout
* 返回值：
    * 成功，0；
    * 失败，-1

```c
//从字符串写入
int main()
{
    FILE* fp=fopen("D:/a.txt","r");
    if(!fp)
    {
        return -1;
    }
    
    /*
    char ch[]="你瞅啥瞅你咋地";	
    fputs(ch,fp);	//文件中写入：你瞅啥瞅你咋地
    */
    
    /*
    char ch[]="你瞅啥\n瞅你咋地";	
    fputs(ch,fp);	//文件中写入两行内容，第一行：你瞅啥		第二行：瞅你咋地	
    */
    
    char ch[]="你瞅啥\0瞅你咋地";	
    fputs(ch,fp);	//文件中写入：你瞅啥
    //及与字符串的行写入，因此在字符串中遇到字符串结束符，写入也就在这里停止了
    
    fclose(fp);
    
    return 0;
}
```

```c
//从键盘写入
int main()
{
    FILE* fp=fopen("D:/a.txt","r");
    if(!fp)
    {
        return -1;
    }
    char* p=(char*)malloc(sizeof(char)*1024);
    while(1)
    {
        memset(p,1024,0);

        scanf("%[^\n]",p);		//接收非换行的字符
        getchar();				//把缓冲区的换行符吃掉

        
        //也可以使用 fgets函数进行输入
        
        //停止输入的命令
        if(!strcmp(p,"comm=exit")
           break;
           
        strcat(p,'\n');
        fputs(p,fp);
    }
    
    free(p);
    fclose(fp);
    
    return 0;
}
```



2）读文件：fgets

```c
# include<stdio.h>
char *fgets(char *s, int size,FILE *stream);
```

* 功能：从 stream 指定的文件内读入字符，保存到所指定的内存空间，直到出现换行字符、读到文件末尾或是读了 size-1 个字符为止，最后会自动加上 '\0' 字符作为结束标志。
* 参数：
    * s，字符串；
    * size，指定最大读取字符串的长度（size-1）
    * stream：文件指针，如果读取键盘输入的字符串，固定写作 stdin
* 返回值：
    * 成功：成功读取的字符串
    * 读到文件尾或者出错：NULL

```c
int main()
{
    FILE* fp=fopen("D:/a.txt","r");
    if(!fp)
    {
        return -1;
    }
    
    char* p=(char*)malloc(sizeof(char)*100);
    memset(p,0,100);
    fgets(p,100,fp);	//如果一次读取的大小小于一行，下次就会接着这里继续读
    
    printf("%s\n",p);
    
    free(p);
    fclose(fp);
    
    return 0;
}
```

```c
int main()
{
    FILE* fp=fopen("D:/a.txt","r");
    if(!fp)
    {
        return -1;
    }
    
    char* p=(char*)malloc(sizeof(char)*100);
    while(!feof(fp))
    {
        memset(p,100,0);
        fgets(p,100,fp);
        
        printf("%s",p);
    }
    
    free(p);
    fclose(fp);
    
    return 0;
}
```

#### 3.4、四则运算

```c
//在文件中生成 100 到 10以内的四则运算题目
enum opt
{
    add,sub,mul,dive
};
int main()
{
    srand((size_t)time(NULL));
	FILE* fp=fopen("D:/四则运算.txt","w");
    if(!fp)
        return -1;
    int a,b;
    char c;
    char* p=(char*)malloc(sizeof(char)*20);
    for(int i=0;i<100;i++)
    {
        a=rand()%10+1;
        b=rand()%10+1;
        switch(rand()%4)
        {
            case add:
                c='+';
                break;
            case sub:
                c='-';
                break;
            case mul:
                c='*';
                break;
            case dive:
                c='/';
                break;
        }
        memset(p,20,0);
        sprintf(p,"%d %c %d = ",a,c,b);
        fputs(p,fp);
    }
    
    free(p);
    p=NULL;
    fclose(fp);
    fp=NULL;
    
    return 0;
}
```

```c
//读取上面生成的四则运算题目的文件，对其进行计算，并存放到另一个文件中
enum opt
{
    add,sub,mul,dive
};
int main()
{
	FILE* fp1=fopen("D:/四则运算.txt","r");
    FILE* fp2=fopen("D:/四则运算结果.txt","w");
    if(!fp1||fp2)
        return -1;
    
    // !feof(fp)		EOF -1
    for(int i=0;i<100;i++)
    {
        char* p=(char*)malloc(sizeof(char)*20);
        memset(p,20,0);
        fgets(p,20,fp);
        int a,b;
    	char c;
        
        sscanf(p,"%d %c %d = ",&a,&cc,&b);
        int sum=0;
        switch(c)
        {
            case '+':sum=a+b;break;
            case '-':sum=a-b;break;
            case '*':sum=a*b;break;
            case '/':sum=a/b;break;
        }
        memset(p,0,20);
        sprintf(p,"%d %c %d = %d\n",a,c,b,sum);
        fputs(p,fp);
    }
    
    free(p);
    p=NULL;
    fclose(fp);
    fp=NULL;
    
    return 0;
}
```

#### 3.4、按照格式化文件 fprintf、fscanf

1）写文件

```c
#include<stdio.h>
int fprintf(FILE* stream,const char* format,…);
```

* 功能：根据参数 format 字符串来转换并格式化数据，然后将结果输出到 stream 指定的文件中，指定出现字符串结束符 \0 为止
* 参数：
    * stream：已经打开的文件
    * format：字符串格式，用法和 printf 一致
* 返回值：
    * 成功：实际写入文件的字符个数
    * 失败：-1

```c
//字符串读取
int main()
{
    FILE* fp=fopen("D:/a.txt","r");
    if(!fp)
        return -1;
    
    char* p=(char*)malloc(sizeof(char)*1024);
    fprintf(fp,"%s",p);
    
    int a=10;
    int b=2;
    int c=12;
    fprintf(fp,"%05d + %05d = %05d\n",a,b,c);		//在文件中写入了：00010 + 00002 = 00012
    
    free(p);
    fclose(fp);
    return 0;
}
```



2）读文件

```c
#include<stdio.h>
int fscanf(FILE* stream,const char* format,…);
```

* 功能：从 stream 指定的文件读取字符串，并根据参数 format 字符串转换并格式化数据
* 参数：
    * stream：已经打开的文件
    * format：字符串格式，用法和 scanf 一致
* 返回值：
    * 成功：成功转换的值的个数
    * 失败：-1

```c
//字符串读取
int main()
{
    FILE* fp=fopen("D:/a.txt","r");
    if(!fp)
        return -1;
    
    char* p=(char*)malloc(sizeof(char)*1024);
    fscanf(fp,"%s",p);
    printf("%s\n",p);		//格式化读取过程中，通过 %3s，%5c 等方式，可以限定读取的宽度
    //如果使用 %5c 的方式限定读取宽度，需要先初始化字符数组为 0，这样可以在读取结束之后，在末尾自动有字符串结束符，可以直接以字符串形式输出
    
    fscanf(fp,"%s",p);
    printf("%s\n",p);		//fscanf 在读取过程中，遇到空格和回车就结束读取，因此无法读取到空格和换行
    
    free(p);
    fclose(fp);
    return 0;
}
```

```c
//数据读取
int main()
{
    FILE* fp=fopen("D:/a.txt","r");
    if(!fp)
        return -1;
    
    int a,b,c;
    
    /*
    fscanf(fp,"%d",&a);		//格式化读取文件中的数据
    printf("%d\n",a);
    */
    //文件中在第一行写的是 3*6=18
    fscanf(fp,"%d*%d=%d",&a,&b,&c);		//格式化读取文件中的数据
    printf("%d\n%d\n%d\n",a,b,c);		//输出：3,6,18
    //读取过程中，如果是以十进制方式读取，那么遇到非十进制数就停止，
    //同样地，对于八进制和十六进制也是读到非八进制数字和非十六进制数字就停止了
    
    free(p);
    fclose(fp);
    return 0;
}
```

```c
//四则运算
int main()
{
	FILE* fp=fopen("D:/四则运算.txt","w");
    if(!fp)
        return -1;
    
    srand((size_t)time(NULL));
    
    int a,b;
    char c;
    for(int i=0;i<100;i++)
    {
        a=rand()%10+1;
        b=rand()%10+1;
        switch(rand()%4)
        {
            case 0:c='+';break;
            case 1:c='-';break;
            case 2:c='*';break;
            case 3:c='/';break;
        }
        fprintf(fp,"%d %c %d =\n",a,c,b);
    }
    
    fclose(fp);
	return 0;
}
```

#### 3.5、大文件数据排序

```c
//大文件数据排序，生成数据
int main()
{
	FILE* fp=fopen("D:/数据.txt","w");
    if(!fp)
        return -1;
    srand((size_t)time(NULL));
    
    //填入随机数
    for(i=0;i<10000;i++)
    {
        fprintf(fp,"%d\n",rand()%1024);
    }
    
    
    fclose(fp);
	return 0;
}
```

```c
//大文件数据排序，数据排序
//冒泡排序
void Bubble(int* src,int len)
{}

//哈希排序
void hashSort(const int* arr,int* sortedArr,int len)
{
    for(int i=0;i<1000;i++)
    {
        sortedArr[arr[i]]++;
    }
}

int main()
{
	FILE* fp1=fopen("D:/数据.txt","r");
    FILE* fp2=fopen("D:/数据排序.txt","w");
    if(!fp1||!fp2)
        return -1;
    
    //冒泡排序
   	int* arr=(int*)malloc(sizeof(int)*10000);
    for(int i=0;i<10000;i++)
    {
        fscanf(fp1,"%d\n",&arr[i]);
    }
    /*
    Bubble(arr,10000);
    for(int i=0;i<10000;i++)
    {
        fprintf(fp2,"%d\n",arr[i]);
    }
    */
    int sortedArr[1024]={0};
    hashSort(ar,sortedArr,10000);
    for(int i=0;i<1024;i++)
    {
        for(int j=0;j<sortedArr[i];j++)
        {
            fprintf(fp2,"%d\n",i);
        }
    }
    
    fclose(fp1);
    fclose(fp2);
    free(arr);
	return 0;
}
```

#### 3.6、按照块读写文件 fread、fwrite（针对二进制文件的操作）

1）写文件

```c
#include<stdio.h>
size_t fwrite(const void* ptr,size_t size,size_t nmemb,FILE* stream);
```

* 功能：以数据块的方式给文件写入内容
* 参数：
    * ptr：准备写进文件的数据地址
    * size：指定写入文件内容的块数据的大小
    * nmemb：写入文件的块数        写入文件的数据总大小：size*nmemb
    * stream：已经打开的文件指针
* 返回值：
    * 成功：实际成功写入文件数据的块数，此值和 nmemb 相等
    * 失败：0

```c
int main()
{
	FILE* fp=fopen("D:/a.txt","wb");
    if(!fp)
        return -1;
    int a=5678;
    fwrite(&a,sizeof(int),1,fp);	//以二进制形式在文件中存入了 5678，占 4 个字节
    fclose(fp);
    fp=NULL;
    
    FILE* fp1=fopen("D:/a.txt","rb");
    int value;
    fread(&value,sizeof(int),1,fp);
    printf("%d\n",value);		//5678
    fclose(fp1);
    fp1=NULL;
    
	return 0;
}
```

```c
int main()
{
	FILE* fp=fopen("D:/a.txt","wb");
    if(!fp)
        return -1;
    int arr[]={1,2,3,4,5,6,7,8,9,10};
    fwrite(arr,sizeof(int),10,fp);	
    fclose(fp);
    fp=NULL;
    
    FILE* fp1=fopen("D:/a.txt","rb");
    int value[]={0};
    fread(value,sizeof(int),10,fp);		//写成：fread(value,10,4,fp);	也可以，其他写法也都可以，保证大小总一致就可以
    for(int i=0;i<10;i++)
    {
        printf("%d\n",value[i]);	//1 2 3 4 5 6 7 8 9 10
    }
    fclose(fp1);
    fp1=NULL;
    
	return 0;
}
```

```c
//文件读写结构体
typedef struct student
{
    char name[21];
    int age;
    int score;
    char addr[51];
}stu;

int main()
{
    stu ss[3]=
    {
        {"单据",9,10,"附件为废物"},
        {"层面上年纪",2,110,"附吃什么词废物"},
        {"层面上墨",90,120,"VN无法未付款"}
    };
    FILE* fp=fopen("D:/a.txt","wb");
    if(!fp)
        return -1;
    for(int i=0;i<3;i++)
    {
        fwrite(&ss[i],sizeof(stu),1,fp);
    }
    fclose(fp);
    
    stu* ss=(stu*)malloc(sizeof(stu)*3);
    FILE* fp1=fopen("D:/a.txt","rb");
    if(!fp1)
        return -1;
    
    while(!feof(fp1))
    {
        int i=0;
        fread(ss+i,sizeof(stu),1,fp1);
        i++;
        
    }
    for(int i=0;i<3;i++)
    {
        printf("姓名：%s\n"，ss[i].name);
        printf("年龄：%d\n"，ss[i].age);
        printf("成绩：%d\n"，ss[i].score);
        printf("地址：%s\n"，ss[i].addr);
        
    }
    
    free(ss);
    fclose(fp1);
    return 0;
}
```

2）读文件

```c
#include<stdio.h>
size_t fread(const void* ptr,size_t size,size_t nmemb,FILE* stream);
```

* 功能：以数据块的方式从文件中读取内容
* 参数：
    * ptr：准备读入文件数据的变量地址
    * size：指定读取文件内容的块数据的大小
    * nmemb：读取文件的块数        读取文件的数据总大小：size*nmemb
    * stream：已经打开的文件指针
* 返回值：
    * 成功：实际成功读取文件数据的块数，此值和 nmemb 相等
    * 失败：0

#### 3.7、大文件拷贝

```c
#define SIZE 1024
int main(int argc,char* argv[])
{
    for(int i=0;i<argc;i++)
    {
        printf("%s\n",arfv[i]);
    }
    
    //用户输入参数缺少
    if(argc<3)
    {
        printf("缺少参数\n");
        return -1;
    }
    
    //argv[0] 程序名称 文件大小50MB
    //D:/copy.exe D:/test.avi D:/code/test.avi
    FILE* fp1=fopen(argv[1],"rb");
    FILE* fp2=fopen(argv[2],"wb");
    
    if(!fp1||!fp2)
    {
        printf("复制文件出错\n");
        return -2;
    }
    
    char* temp=(char*)malloc(sizeof(char)*SIZE);
    int count=0;
    while(!feof(fp1))
    {
        memset(temp,0,SIZE);
        //fread(temp,sizeof(char),SIZE,fp1);
       // fwrite(temp,sizeof(char),SIZE,fp2);		//拷贝完成后，文件大小多了几十字节
    //原因是，最后一次拷贝，可能剩余的文件内容小于1024字节，实际还是拷贝了1024的大小，所以拷贝后更大了
        //可以把写入文件的操作写成这样来解决这个问题：
        count=fread(temp,sizeof(char),SIZE,fp1);		//可以修改每次拷贝的数据块的大小到合适的值，使拷贝速度更快
        fwrite(temp,sizeof(char),count,fp2);
    }
    free(temp);
    fclose(fp1);
    fclose(fp2);		
    
    return 0;
}
```

```c
#include<sys/types.h>
#include<sys/stat.h>

#define SIZE 1024*1024*8		//设置单次拷贝数据块大小为8MB
int main(int argc,char* argv[])
{
    for(int i=0;i<argc;i++)
    {
        printf("%s\n",arfv[i]);
    }
    
    //用户输入参数缺少
    if(argc<3)
    {
        printf("缺少参数\n");
        return -1;
    }
    
    //argv[0] 程序名称 文件大小50MB
    //D:/copy.exe D:/test.avi D:/code/test.avi
    FILE* fp1=fopen(argv[1],"rb");
    FILE* fp2=fopen(argv[2],"wb");
    
    if(!fp1||!fp2)
    {
        printf("复制文件出错\n");
        return -2;
    }
    char* temp=NULL;
    int realSize=0;
    struct stat st;
    stat(argv[1],st);
    if(st.st_size>SIZE)		//根据文件实际大小开辟空间
    {
        temp=(char*)malloc(sizeof(char)*SIZE);
        realSize=SIZE;
    }
    else
    {
        temp=(char*)malloc(sizeof(char)*(st.st_size+10));
        realSize=st.st_size+10
    }
    
    int count=0;
    while(!feof(fp1))
    {
        memset(temp,0,realSize);
        count=fread(temp,sizeof(char),realSize,fp1);		//可以修改每次拷贝的数据块的大小到合适的值，使拷贝速度更快
        fwrite(temp,sizeof(char),count,fp2);
    }
    free(temp);
    fclose(fp1);
    fclose(fp2);		
    
    return 0;
}
```



### 4、获取文件的状态

```c
#include<sys/types.h>
#include<sys/stat.h>
int stat(const char* path,struct stat* buf);
```

* 功能：获取文件状态信息
* 参数：
    * path：文件名
    * buf：保存文件信息的结构体
* 返回值：
    * 成功：0
    * 失败：-1

```c
//文件信息结构体
struct stat
{
    dev_t st_dev;				//文件的设备编号
    ino_t st_ino;				//节点
    mode_t st_mode;				//文件的类型和存取的权限
    nlink_t st_nlink;			//链接到该文件的硬链接的数目，刚建立的文件的该值为 1
    uid_t st_uid;				//用户 ID
    gid_t st_gid;				//组 ID
    dev_t st_rdev;				//(设备类型)若此文件为设备文件，则为其设备编号
    off_t st_size;				//文件字节数（文件大小）
    unsigned long st_blksize;	//块大小（文件系统的 I/O 缓冲区大小）
    unsigned long st_blocks;	//块数
    time_t st_atime;			//最后一次访问时间
    time_t st_mtime;			//最后一次修改时间
    time_t st_ctime;			//最后一次改变时间（指属性）
};
```

```c
#include<sys/types.h>
#include<sys/stat.h>
//获取文件状态
int main()
{
    ////文件信息结构体变量
    struct stat st;
    stat("D:/copy.eex",&st);
    printf("文件大小：%d\n"，st.st_size);		//输出文件的大小（单位：字节）
    
    return 0;
}
```

### 5、文件的随机读写

1）fseek() 函数

```c
#include<stdio.h>
int fseek(FILE* stream,long offset,int whence);
```

* 功能：移动文件流（文件光标）的读写位置
* 参数：
    * stream：已经打开的文件指针
    * offset：根据 whence 来移动的位数（偏移量），可以是正数，亦可以是负数。正数表示相对于 whence 往右移动，负数表示相对于 whence 往左移动。如果向左移动的字节数超过了文件开头则报错返回，如果向右移动的字节数超过了文件末尾则再次写入时将增大文件大小
    * whence，其取值如下：
        * SEEK_SET：从文件开头移动 offset 个字节（宏定义值为 0）
        * SEEK_CUR：从当前位置移动 offset 个字节（宏定义值为 1）
        * SEEK_END：从文件末尾移动 offset 个字节（宏定义值为 2）
* 返回值：
    * 成功：0
    * 失败：-1

```c
//文件随机读写

/*
a.txt 内容：
123456789
你瞅啥
瞅你咋地
再瞅一个试试
对不起大哥 我错了
*/
int main()
{
    FILE* fp=fopen("D:/a.txt","wb");
    if(!fp)
        return -1;
    char arr[100];
    memset(arr,0,100);
    
    fgets(arr,100,fp);
    printf("%s\n",arr);		//输出：123456789
    fgets(arr,100,fp);
    printf("%s\n",arr);		//输出：你瞅啥
    
    //文件随机读写
    fseek(fp,-8,SEEK_CUR);
    fgets(arr,100,fp);
    printf("%s\n",arr);		//输出：你瞅啥
    
    //从文件起始位置开始偏移
    fseek(fp,11,SEEK_SET);
    fgets(arr,100,fp);
    printf("%s\n",arr);		//输出：你瞅啥
    
    ////从文件末尾位置开始偏移
    fseek(fp,-17,SEEK_END);
    fgets(arr,100,fp);
    printf("%s\n",arr);		//输出：对不起大哥 我错了
    
    fclose(fp);
    return 0;
}
```



2）ftell() 函数

```c
#include<stdio.h>
long ftell(FILE* stream);
```

* 功能：获取文件流的读写位置
* 参数：stream，打开的文件指针
* 返回值：
    * 成功：当前文件流的读写位置
    * 失败：-1

```c
//在文件中间指定位置添加内容
/*
a.txt 内容：
123456789
你瞅啥
瞅你咋地
再瞅一个试试
对不起大哥 我错了
*/
int main()
{
    FILE* fp=fopen("D:/a.txt","r+");	//使用 "a" 的模式打开，存在缓冲区的问题，导致使用 fseek 操作文件流可能会不起作用，最终添加的内容还是添加在文件末尾
    if(!fp)
        return -1;
    fseek(fp,-17,SEEK_END);
    long pos=ftell(fp);
    printf("%ld\n",p)
    fputs("瞅你咋地\n",fp);
    
    fclose(fp);
    /*文件中的内容修改后为：
    123456789
	你瞅啥
	瞅你咋地
	再瞅一个试试
	瞅你咋地
	对不起大哥 我错了
    */
    return 0;
}
```



3）rewind() 函数

```c
#include<stdio.h>
void rewind(FILE* stream);
```

* 功能：把文件流的读写位置移动到文件开头
* 参数：stream，打开的文件指针
* 返回值：无返回值

```c
frewind(fp);		//与这个语句效果一样：fseek(fp,0,SEEK_SET);
```

### 6、删除文件、重命名文件

1）删除文件

```c
#include<stdio.h>
int remove(const char* pathname);
```

* 功能：删除文件（直接删除了，不会在回收站中出现）
* 参数：pathname，文件名
* 返回值：
    * 成功：0
    * 失败：-1

2）重命名文件

```c
#include<stdio.h>
int rename(const char* oldpath,const char* newpath);
```

* 功能：把 oldname 重命名为 newpath（还具有移动文件的功能，如果新旧文件的路径不同，就直接把文件移动了）
* 参数：
    * oldpath：旧文件名
    * newpath：新文件名
* 返回值：
    * 成功：0
    * 失败：-1

```c
int main()
{
    int value=remove("D:/b.txt");
    if(!value)
        printf("删除成功\n");
    else
        printf("删除失败\n");
    
    int value1=rename("D:/b.txt","D:/abc.txt");
    if(!value1)
        printf("重命名成功\n");
    else
        printf("重命名失败\n");
    
    return 0；
}
```

### 7、文件缓冲区

#### 7.1、文件缓冲区介绍

* 缓冲文件系统是系统自动地在内存区为程序中每一个正在使用的文件开辟一个文件缓冲区，从内存向磁盘输出的数据必须先送到内存中的缓冲区，装满缓冲区后才一起送到磁盘中去；
* 从磁盘向计算机读入数据，则一次从磁盘将一批数据输入到内存缓冲区（充满缓冲区），再从缓冲区逐个地将数据送到程序数据区（给程序变量）。
* 缓冲区是内存的一部分，断电丢失

#### 7.2、磁盘文件的存取

* 磁盘文件，一般保存在硬盘、U盘等掉电不丢失的磁盘设备中，在需要时调入内存
* 在内存中对文件进行编辑处理后，保存到磁盘中
* 程序与磁盘之间交互，不是立即完成，系统或程序可以根据需要设置缓冲区，以提高存储效率

#### 7.3、更新缓冲区

```c
#include<stdio.h>
int fflush(FILE* stream);
```

* 功能：更新缓冲区，让缓冲区数据立马写到文件中
* 参数：stream，文件指针
* 返回值：
    * 成功：0
    * 失败：-1

```c
int main()
{
    FILE* fp=fopen("D:/a.txt","w+");
    if(!fp)
        return -1;
    char ch;
    while(1)
    {
        scanf("%c",&ch);
        if(ch=='@')
            break;
        fputc(ch,fp);
        //频繁地个硬盘交互，损伤硬盘，不建议频繁地更新缓冲区
        fflush(fp);
    }
    
    fclose(fp);
    return 0;
}
```

### 8、实战项目-快译通

#### 8.1、核心代码

**dict.c：** 

```c
#include<stdlib.h>
#include<math.h>
#include<time.h>
#include<stdio.h>
#include<dict.h>
#define SIZE 3

/*
1、创建结构体，存储单词和解释
2、读取单词，格式化存储对应的堆空间中
3、单词查找
4、销毁堆空间
*/

/*
单词库文件，dict.txt:
#a
Trans:art. 一；字母A
#a.m.
Trans:n. 上午
#a/c
Trans:n. 往来账户@往来：come - and - go；contact；intercourse@n. 往来账户
*/

int getWord()
{
    FILE* fp=fopen("D:/dict.txt","r");
    if(!fp)
    {
        printf("加载单词库失败\n");
        return -1;
    }
    list=(dict*)malloc(sizeof(dict)*SIZE);
    
    index=(pos*)malloc(sizeof(pos)*27);
    char flag='a';	//记录当前索引标志位
    int idx=0;		//记录对应字母的索引
    index[0].start=0;	//记录 a 的索引
    index[0].end=0;
    
    //数组下标
    int i=0;
    char* temp=(char*)malloc(sizeof(char)*1024);
    while(!feof(fp))
    {
        memset(temp,0,1024);
        fgets(temp,1024,fp);
        
        //去掉单词末尾的换行符
        temp[strlen(temp)-1]=0;
        
        //开辟单词的堆空间
        list[i].word=(char*)malloc(sizeof(char)*(strlen(temp)));
        //将单词放在指定的堆空间中
        //从第二个字符开始存放，去掉开头的 #
        strcpy(list[i].word,&temp[1]);
        
        //0-25 分别存放不同字母开头的单词
        //0-25 index[0].start	index[0].end
        //创建索引
        if(idx!=26)
        {
            if(list[i].word[0]==flag)
        	{
            	index[idx].end++;
            
        	}
        	else
        	{
            	idx++;
            	index[idx].start=index[idx-1].end;
            	index[idx].end=index[idx-1].end;
            	flag++;
        	}
        }
        
        
        //去掉翻译末尾的换行符
        temp[strlen(temp)-1]=0;
        
        memset(temp,0,1024);
        fgets(temp,1024,fp);
        list[i].trans=(char*)malloc(sizeof(char)*(strlen(temp)-5));		//开辟内存的时候去掉前面多的内容，不需要存储
        //从第七个字符开始存放，去掉开头的 Trans:
        strcpy(list[i].trans,&temp[6]);
        
        //输出
        printf("%s\n",list[i].word);
        printf("%s\n",list[i].trans);
        
        i++;
    }
    //记录中文的索引
    index[26].start=index[25].end;
    index[26].end=SIZE;
    
    //释放堆空间和关闭文件
    free(temp);
    fclose(fp);
    
    return i;
}

int searchWord(const char* word,const char* trans,int idx)
{
    if(!word||!trans)
    {
        printf("输入异常\n");
        return -1;
    }
    
    for(int i=index[idx].start;i<index[idx].end,i++)
    {
        if(!strcmp(word,list[i].word))	//相同
        {
            strcpy(trans,list[i].trans);
            return 0;
        }
    }
    return 1;
}

//数据销毁
void destroySpace()
{
    if(!list)
        return;
    if(index)
    {
        free(index);
        index=NULL;
    }
    for(int i=0;i<SIZE;i++)
    {
        free(list[i].word);
        free(list[i].trans);
        
    }
    free(list);
    list=NULL;
}


int main()
{
    //获取单词库
    int wordCnt=getWord();
    
    //接收用户输入的单词
    char* word=(char*)malloc(sizeof(char)*1024);
    //根据单词提供的翻译
    char* trans=(char*)malloc(sizeof(char)*1024);
    int idx=0;
    while(1)
    {
        memset(word,0,1024);
        memset(trans,0,1024);
        //scanf("%s",word);
        //scanf("%[^\n]",word);
        gets(word);
        
        //出口
        if(!strcmp(word,"comm=exit"));
        
        //0-26
        if(*word>='a'&&*word<='z')
        {
            idx=*word-'a';
        }
        else
        {
            idx=26;
        }
        if(!searchWord(word,trans,idx))
        {
            printf("%s\n",trans);
        }
        else
        {
            printf("未找到该单词\n");
        }
        
    }
    
    //销毁空间
    free(word);
    free(trans);
    destroySpace();
    
    return 0;
}
```

**dict.h：**

```c
#pragma once
#include<stdlib.h>
typedef struct DICT
{
    char* word;
    char* trans;
}dict;

typedef struct POSTION
{
	int start;
    int end;
}pos;

//记录单词的下标
pos* index=NULL;

dict* list=NULL;

//函数的声明

//获取单词库
int getWord();

//查找单词
int searchWord(const char* word,const char* trans,int idx);

//销毁空间
void destroySpace();
```

#### 8.2、索引

1）单词库文件行数

```c
int main()
{
    FILE* fp=fopen("D:/dict.txt","r");
    if(!fp)
    {
        return -1;
    }
    
    char* p=(char*)malloc(sizeof(cahr)*1024);
    int i=0;
    while(!feof(fp))
    {
        fgets(p,1024,fp);
        i++;
    }
    printf("%d\n",i);
    fclose(fp);
    return 0;
}
```

在快译通中添加每个字母开头的单词的索引，加快搜索速度。