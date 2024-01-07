---
layout: post
title: c_advance_3_file_api
date: 2023-12-28
tags: [c_cpp]
author: taot
---

## 文件api

### 1、文件基本操作

#### 1.1、文件操作的步骤

* 引入头文件（stdio.h）
* 打开文件
* 使用文件指针
* 关闭文件

#### 1.2、文件相关的概念

* 按文件的逻辑结构：记录文件、流式文件
* 按存储介质：普通文件、设备文件
* 按数据的组织形式：文本文件、二进制文件

#### 1.3、文件缓冲区

* 在程序的数据区和磁盘之间进行数据交换的时候，要经过缓冲区的传递
* 不同平台的缓冲区大小不同
* 刷新缓冲区：fflush(fp)
* 在文件指针关闭、程序结束、文件缓冲区满的三种情况下，缓冲区的内容都会写入文件，否则不会
* 对于 ACSI C 标准采用文件缓冲区，系统调用不会使用文件缓冲区

#### 1.4、输入输出流

* 流表示了信息从源到目的端的流动
* 输入操作时：数据从文件流向计算机内存（文件读取）
* 输出操作时：数据从计算机六流向文件（文件的写入）
* 文件句柄：实际是一个结构体，标志着文件的各种状态

#### 1.5、文件操作 API

* fgets、fputc：按照字符读写文件
* fputs、fgets：按照行读写文件（读写配置文件）
* fread、fwrite：按照块读写文件（大数据块迁移）
* fprintf、fscanf：按照格式化进行读写文件

### 2、标准的文件读写

#### 2.1、文件的顺序读写

```c
typedef struct Stu
{
    char name{50};
    int id;
}Stu;

int main()
{
    fputc('c',stdio);		//字符输出到标准输出，此时没有缓冲区，缓冲区是针对普通文件的
    char ch=fgetc(stdin);	
    fputc(stderr,"%c",ch);	//标准错误输出，//stderr 通常也是指向屏幕的
    
    FILE *fp=NULL;
    //绝对路径、相对路径（./test.txt、../test.txt）
    //直接运行可执行程序，相对路径是相对于可执行程序的
    //字符串续行符：
    /*
    char *p="aacscscscsdscaxasca"\
    "cscnsjcnkcskc";
    */ //表示一个字符串
    
    // "w+"，写读的方式打开。如果文件不存在，则创建文件；如果文件存在，则清空文件内容，再写
    fp=fopen("文件路径/test.txt","w+");
    if(fp==NULL)
    {
        perror("fopen");		//提示错误信息，字符串的内容会在错误信息前面输出，作为错误信息的标识
        return -1;
    }
    
    //按字符写文件
    char buf="cnsakcsccmdmaasx";
    int i=0;
    int n=strlen(buf);
    for(i=0;i<n;i++)
    {
        int res=fputc(buf[i],fp);			//返回值是成功写入文件的字符的 ASCII 码
        //写完之后，会自动添加添加文件结束符
    }
    //文件关闭
    if(fp!=NULL)
    {
        fclose(fp);
        fp=NULL;
    }
    
    
    
    //按字符读文件
    FILE *fp=NULL;
    //"r+"，文件以读写方式打开，如果文件不存在，打开失败
    fp=fopen("./test.txt","r+");
    if(fp==NULL)
    {
        perror("fopen");		//提示错误信息，字符串的内容会在错误信息前面输出，作为错误信息的标识
        return -1;
    }
    char ch;
    while(ch=fgetc(fp)!=EOF)		//等价于：while(!feof(fp)){ch=fgetc(fp);printf("%c",ch)}
    {
        printf("%c",ch);
    }
    printf("\n");
    if(fp!=NULL)
    {
        fclose(fp);
        fp=NULL;
    }
    
    
    //按照行读写文件
    FILE *fp=NULL;
    fp=fopen("文件路径/test.txt","w+");
    if(fp==NULL)
    {
        perror("fopen");		//提示错误信息，字符串的内容会在错误信息前面输出，作为错误信息的标识
        return -1;
    }
    char *buf[]={"aaaaaaaaaa\n","ssssssssss\n","dddddddddd\n","ffffffffff\n"};
    for(int i=0;i<4;i++)
    {
        int res=fputs(buff[i],fp);		//返回值：0代表成功，非零代表失败
    }
    
    char rbuf[30]={0};
    while(!feof(fp))
    {
        //sizeof(buf),表示最大读取的字节数。如果文件当行字节数大于sizeof(buf)，则只读取前sizeof(buf)；若文件件当行字节数小于sizeof(buf)，则按照当行实际长度全部读取
        //返回值是成功读取的文件内容
        //读取的过程中，以 "\n"作为一行结束的标识
        //fgets 读取完毕后，还自动在读取的字符串末尾添加字符串结束符
        char *p=fgets(rbuf,sizeof(buf),fp);		
        printf("%s\n",rbuf);
    }
    if(fp!=NULL)
    {
        fclose(fp);
        fp=NULL;
    }
    
    
    
    //按照块写文件
    FILE *fp=NULL;
    fp=fopen("文件路径/test.txt","w+");
    if(fp==NULL)
    {
        perror("fopen");		//提示错误信息，字符串的内容会在错误信息前面输出，作为错误信息的标识
        return -1;
    }
    Stu s[3];
    char buf[50];
    for(int i=0;i<3;i++)
    {
        sprintf(buf,"stu%d%d%d",i,i,i);
        strcpy(s[i].name,buf);
        s[i].id=i+1;
    }
    //按块写文件
    //s，写入文件的内容的首地址
    //sizeof(Stu)，按块写文件的单块的大小
    //3，块数，写文件数据的大小：sizeof(Stu)*3
    //fp，文件指针
    //返回值：成功写入文件的块数
    //fwrite 写入默认是按照二进制进行
    int res=fwrite(s,sizeof(Stu),3,fp);		//如果三块都写入成功，res=3
    if(fp!=NULL)
    {
        fclose(fp);
        fp=NULL;
    }
    
    
    
    
    //按照块读文件
    FILE *fp=NULL;
    fp=fopen("文件路径/test.txt","r+");
    if(fp==NULL)
    {
        perror("fopen");		//提示错误信息，字符串的内容会在错误信息前面输出，作为错误信息的标识
        return -1;
    }
    Stu s[3];
    char buf[50];
    //按块读文件
    //s，放文件内容的首地址
    //sizeof(Stu)，按块读文件的单块的大小
    //3，块数，读文件数据的大小：sizeof(Stu)*3
    //fp，文件指针
    //返回值：成功读取文件的块数
    //fread 读取默认是按照二进制进行
    int ret=fread(s,sizeof(Stu),3,fp);		//如果三块都读取成功，ret=3
    for(int i=0;i<3;i++)
    {
        printf("%s %d\n",s[i].name,s[i].age);
    }
    if(fp!=NULL)
    {
        fclose(fp);
        fp=NULL;
    }
    
    
    
    
    // 按照格式化写文件
    FILE *fp=NULL;
    fp=fopen("文件路径/test.txt","w+");
    if(fp==NULL)
    {
        perror("fopen");		//提示错误信息，字符串的内容会在错误信息前面输出，作为错误信息的标识
        return -1;
    }
    printf("hello\n");	//等价于：fprintf(stdout,"hello");
    fprintf(fp,"hello,	I am pig,mike=%d\n",250);	//文件中的内容：hello,	I am pig,mike=250
    if(fp!=NULL)
    {
        fclose(fp);
        fp=NULL;
    }
    
    
    
    //格式化读取文件
    FILE *fp=NULL;
    fp=fopen("文件路径/test.txt","w+");
    if(fp==NULL)
    {
        perror("fopen");		//提示错误信息，字符串的内容会在错误信息前面输出，作为错误信息的标识
        return -1;
    }
    //printf("hello\n");	//等价于：fprintf(stdout,"hello");
    //fprintf(fp,"hello,I am pig,mike=%d\n",250);	//文件中的内容：hello,	I am pig,mike=250
    int a=0;
    fscanf(fp,"hello,I am pig,mike=%d\n",&a);
    printf("%d\n",a);		//250
    if(fp!=NULL)
    {
        fclose(fp);
        fp=NULL;
    }
    
    
    
    return 0;
}
```

#### 2.2、文件的随机读写

```c
#include<stdio.h>

int fseek(FILE *stream,long offset,int whence);
//返回值：成功返回0；错返回-1，并设置errno
//offset，偏移量，可正可负可为零
//whence，光标开始移动的起点

long ftell(FILE *stream);
//返回值：成功返回当前读写位置；出错返回-1，并设置errno

void rewind(FILE *stream);
//该函数是把光标移动文件开头
```

```c
int main()
{
    //随机位置读文件    
    FILE *fp=NULL;
    fp=fopen("文件路径/test.txt","r+");			//文件中存放了前面那个结构体数组的三个元素
    if(fp==NULL)
    {
        perror("fopen");		//提示错误信息，字符串的内容会在错误信息前面输出，作为错误信息的标识
        return -1;
    }
    Stu s[3];
    Stu tmp;
    //读第三个结构体
    fseek(fp,sizeof(Stu)*2,SEEK_SET);		//等价于：fseek(fp,sizeof(Stu)*(-1),SEEK_END);
    int ret=fread(&tmp,sizeof(Stu),1,fp);
    if(ret==1)
    {
        printf("%s %d\n",tmp.name,tmp.id);
    }
    //把光标移动到最开始的地方，读取前两个
    fseek(fp,0,SEEK_SET);			//等价于：rewind(fp);
    ret=fread(s,sizeof(Stu),2,fp);
    for(int i=0;i<2;i++)
    {
        printf("%s %d\n",s[i].name,s[i].id);
    }
    if(fp!=NULL)
    {
        fclose(fp);
        fp=NULL;
    }
    return 0;
}
```