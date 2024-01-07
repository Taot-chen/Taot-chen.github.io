---
layout: post
title: c_advance_2_struct
date: 2023-12-28
tags: [c_cpp]
author: taot
---

## 结构体

### 1、结构体的基本操作

1）结构体类型的定义

* struct 关键字，struct Teacher 合在一起才是类型
* {} 后面有分号

2）结构体变量的定义

* 先定义类型，再定义变量
* 定义结构体的同时定义结构体变量
* 无名称的结构体类型

3）结构体变量的初始化

* 定义变量是直接初始化，通过 {}

4）typedef 改类型名

```c
typedef struct Teacher
{
    char name[50];
    int age;
}Teacher1;
```

5）点运算符和指针法操作结构体

6）结构体也是一种数据类型，复合类型（自定义类型）

```c
//结构体类型定义
//struct 关键字，struct Teacher 合在一起才是类型
//{} 后面有分号
struct Teacher
{
    char name[50];
    int age;
};
//结构体变量定义，全局变量
struct Teacher t;
int main()
{
    //结构体变量定义，局部变量
    struct Teacher t2={"lili",18};
    printf("%s %d\n",t2.name,t2.age);
    
    strcpy(t2.name,"xiaoming");
    t2.age=22;
    
    struct Teacher *p=NULL;		//指针指向空，无法对其成员进行操作
    p=&t2;					
    strcpy(p->name,"xiaoming");
    p->age=22;			//等价于：(*p).age=22;
    printf("%s %d\n",p->name,p->age);
    
    return 0;
}
```

#### 1.1、结构体内存四区

* 与普通变量的内存四区一致

#### 1.2、结构体变量相互赋值

```c
//定义结构体类型时。不要直接给成员赋值
//结构体只是一个类型，还没有分配空间
//只有根据类型定义变量时，才分配空间，有空间后才能赋值
typedef struct Teacher
{
    char name[50];
    //int age=50;		//err
    int age;
}Teacher;

void copyTeacher(Teacher to,Teacher from)
{
    to=from;
}

void copyTeacher1(Teacher *to,Teacher *from)
{
    *to=*from;
}

int main()
{
    Teacher t1={"lily","22"};
    
    //相同类型的两个结构体变量，可以相互赋值
    //把 t1 成员变量内存的值拷贝给 t2 成员变量的内存
    //t1 和 t2 在内存上没有关系，类似于：int a=10;   int b=a;	
    Teacher t2=t1;
    printf("%s %d\n",t2.name,t2.age);		//lily 22
    
    Teacher t3;
    memset(&t3,0,sizeof(t3));
    copyTeacher(t3,t1);			//t1 拷贝给 t3
    printf("%s %d\n",t3.name,t3.age);		// 0		值传递
    
    copyTeacher1(&t3,&t1);			//t1 拷贝给 t3
    printf("%s %d\n",t3.name,t3.age);		// lily 22		地址传递
    
    return 0;
}
```

#### 1.3、结构体数组

1）结构体静态数组

```c
typedef struct Teacher
{
    char name[50];
    int age;
}Teacher;
int main()
{
    //结构体数组初始化方法1：
    Teacher a[3]=
    {
        {"a",18},  
        {"a",18},  
        {"a",18}
    };
    
    //结构体数组初始化方法2：
    //这种属于是静态结构体数组
    Teacher b[3]={"a",18,"a",18};		//剩下没有赋值的部分会自动初始化为 0 
    for(int i=0;i<3;i++)
    {
        printf("%s %d\n",b[i].name,b[i].age);
    }
    
    return 0;
}
```

2）结构体动态数组

```c
typedef struct Teacher
{
    char name[50];
    int age;
}Teacher;
int main()
{
    //类似于：Teacher p[3];
    Teacher* p=(Teacher*)malloc(3*sizeof(Teacher));
    if(p==NULL)
        return -1;
    char buf[50];
    for(int i=0;i<3;i++)
    {
        sprintf(buf,"name%d%d%d\n",i,i,i);
        strcpy(p[i].name,buf);
        p[i].age=20+i;
    }
    for(int i=0;i<3;i++)
    {
        printf("%s %d\n",p[i].name,p[i].age);
    }
    if(p!=NULL)
    {
        free(p);
        p=NULL;
    }
    
    return 0;
}
```

#### 1.4、结构体一级指针

1）结构体套一级指针

```c
typedef struct Teacher
{
    char *name;
    int age;
}Teacher;
int main()
{
    char* name=NULL;
    //strcpy(name,"lily");		//err，因为 name 指向空，无法对其进行拷贝
    name=(char*)malloc(sizeof(char)*30);
    strcpy(name,"lily");
    if(name!=NULL)
    {
        free(name);
        name=NULL;
    }
    
    //1.
    Teacher t;
    t.name=(char*)malloc(sizeof(char)*30);
    strcpy(t.name,"lily");
    t.age=22;
    printf("%s %d\n",t.name,t.age);		//lily 22
    if(t.name!=NULL)
    {
        free(t.name);
        t.name=NULL;
    }
    
    //2.
    Tearcher *p=NULL;
    p=(Teacher*)malloc(sizeof(Teacher)*1);
    p->name=(char*)malloc(sizeof(char)*30);
    strcpy(p->name,"lily");
    p->age=22;
    printf("%s %d\n",p->name,p->age);		//lily 22
    if(p->name!=NULL)
    {
        free(p->name);
        p->name=NULL;
    }
    if(p!=NULL)
    {
        free(p);
        p=NULL;
    }
    
    //3.
    Teacher *q=NULL;
    q=(Teacher*)malloc(sizeof(Teacher)*3);		//Teacher q[3];
    char buf[30];
    for(int i=0;i<3;i++)
    {
        q[i].name=(char*)malloc(sizeof(char)*30);
        sprintf(buf,"name%d%d%d",i,i,i);
        strcpy(q[i].name,buf);
        q[i].age=20+i;
    }
    for(int i=0;i<3;i++)
    {
        printf("%s %d\n",q[i].name,q[i].age);		
    }
    for(int i=0;i<3;i++)
    {
        if(q[i].name!=NULL)
        {
            free(q[i].name);
            q[i].name=NULL;
        }
    }
    if(q!=NULL)
    {
        free(q);
        q=NULL;
    }
    
    
    return 0;
}
```

#### 1.5、结构体做函数参数

```c
typedef struct Teacher
{
    char *name;
    int age;
}Teacher;
void showTeacher(Taecher *q,int n)
{
    int i=0;
    for(i=0;i<n;i++)
    {
        printf("%s %d\n",q[i].name,q[i].age);		
    }
}

void freeTeacher(Teacher *q,int n)
{
    int i=0;
    for(int i=0;i<3;i++)
    {
        if(q[i].name!=NULL)
        {
            free(q[i].name);
            q[i].name=NULL;
        }
    }
    if(q!=NULL)
    {
        free(q);
        q=NULL;
    }
}

Teacher* getMem(int n)
{
    Teacher *q;
    q=(Teacher*)malloc(sizeof(Teacher)*n);		//Teacher q[3];
    char buf[30];
    for(int i=0;i<n;i++)
    {
        q[i].name=(char*)malloc(sizeof(char)*30);
        sprintf(buf,"name%d%d%d",i,i,i);
        strcpy(q[i].name,buf);
        q[i].age=20+i;
    }
    return q;
}

int getMem1(Teacher **tmp,int n)
{
    if(tmp==NULL)
        return -1;
    Teacher *q;
    q=(Teacher*)malloc(sizeof(Teacher)*n);		//Teacher q[3];
    char buf[30];
    for(int i=0;i<n;i++)
    {
        q[i].name=(char*)malloc(sizeof(char)*30);
        sprintf(buf,"name%d%d%d",i,i,i);
        strcpy(q[i].name,buf);
        q[i].age=20+i;
    }
    *tmp=q;
    
    return 0;
}
int main()
{
    Teacher *q=NULL;
    //q=getMem(3);		//值传递，返回值
    
    int ret=0;
    ret=getMem1(&q,3);		//地址传递
    if(ret!=0)
        return ret;
    
    showTeacher(q,3);
    
    freeTeacher(q,3);
    q=NULL;
    
    
    return 0;
}
```

#### 1.6、结构体套二级指针

```c
typedef struct Teacher
{
    char **stu;		//一个老师有多个学生
}Teacher;

int main()
{
    //1.
    Teacher t;		
    //t.stu[3]
    //char *t.stu[3]
    int n=3;
    int i=0;
    t.stu=(char**)malloc(sizeof(char*)*n);
    for(i=0;i<n;i++)
    {
        t.stu[i]=(char*)malloc(30);
        strcpy(t.stu[i],"lily");
    }
    for(i=0;i<n;i++)
    {
        printf("%s\n",t.stu[i]);
    }
    for(i=0;i<n;i++)
    {
        if(t.stu[i]!=NULL)
        {
            free(t.stu[i]);
            t.stu[i]=NULL;
        }
    }
    if(t.stu!=NULL)
    {
        free(t.stu);
        t.stu=NULL;
    }
        
    
    //2.
    Teacher *p=NULL;
    //p->stu[3]
    p=(Teacher*)malloc(sizeof(Teacher));
    p->stu=(char**)malloc(sizeof(char*)*n);
    for(i=0;i<n;i++)
    {
        p->stu[i]=(char*)malloc(30);
        strcpy(p->stu[i],"lily");
    }
    for(i=0;i<n;i++)
    {
        printf("%s\n",p->stu[i]);
    }
    for(i=0;i<n;i++)
    {
        if(p->stu[i]!=NULL)
        {
            free(p->stu[i]);
            p->stu[i]=NULL;
        }
    }
    if(p->stu!=NULL)
    {
        free(t.stu);
        p->stu=NULL;
    }
    if(p!=NULL)
    {
        free(p);
        p=NULL;
    }
    
    
    //3.
    Teacher *q=NULL;
    //Teacher q[3]
    //q[i].stu[3]
    q=(Teacher*)malloc(sizeof(Teacher)*n);		//Teacher q[3]
    for(i=0;i<n;i++)
    {
        (q+i)->stu=(char**)malloc(n*sizeof(char*));  //char *stu[3];
        for(int j=0;j<3;j++)
        {
            (q+i)->stu[j]=(char*)malloc(30);
            char buf[30];
            sprintf(buf,"name%d%d%d",i,i,i);
            strcpy((q+i)->stu[j],buf);
        }
    }
    for(i=0;i<3;i++)
    {
        for(int j=0;j<3;j++)
        {
            printf("%s\n",(q+i)->stu[j]);
        }
    }
    //free
    for(i=0;i<3;i++)
    {
        for(int j=0;j<3;j++)
        {
            if((q+i)->stu[j]!=NULL)
            {
                free((q+i)->stu[j]);
                (q+i)->stu[j]=NULL;
            }
        }
        if(q[i].stu!=NULL)
        {
            free(q[i].stu);
            q[i].stu=NULL;
        }
    }
    if(q!=NULL)
    {
        free(q);
        q=NULL;
    }
    
    
    return 0;
}
```

* 函数封装

```c
typedef struct Teacher
{
    char **stu;		//一个老师有多个学生
}Teacher;

//n1 老师数量，n2 每个老师的学生数量
int creatTeacher(Teacher **tmp,int n1,int n2)
{
    int i=0;
    if(tmp==NULL)
        return -1;
    
    Teacher *q=(Teacher*)malloc(sizeof(Teacher)*n1);		//Teacher q[3]
    for(i=0;i<n1;i++)
    {
        (q+i)->stu=(char**)malloc(n2*sizeof(char*));  //char *stu[3];
        for(int j=0;j<n2;j++)
        {
            (q+i)->stu[j]=(char*)malloc(30);
            char buf[30];
            sprintf(buf,"name%d%d%d",i,i,i);
            strcpy((q+i)->stu[j],buf);
        }
    }
    //间接赋值
    *tmp=q;
    
    return 0;
}

void showTeacher(Teacher *q,int n1,int n2)
{
    if(q==NULL)
    {
        return;
    }
    for(int i=0;i<n1;i++)
    {
        for(int j=0;j<n2;j++)
        {
            printf("%s\n",(q+i)->stu[j]);
        }
    }
}

void freeTeacher(Teacher **tmp,int n1,int n2)
{
    if(tmp==NULL)
        return;
    Teacher *q=*tmp;
    //free
    for(int i=0;i<n1;i++)
    {
        for(int j=0;j<n2;j++)
        {
            if((q+i)->stu[j]!=NULL)
            {
                free((q+i)->stu[j]);
                (q+i)->stu[j]=NULL;
            }
        }
        if(q[i].stu!=NULL)
        {
            free(q[i].stu);
            q[i].stu=NULL;
        }
    }
    if(q!=NULL)
    {
        free(q);
        q=NULL;
        *tmp=NULL;
    }
}

int main()
{   
    //3.
    Teacher *q=NULL;
    int ret=0;
    ret=creatTeacher(&q,3,3);
    if(ret!=0)
        return -1;
    //Teacher q[3]
    //q[i].stu[3]
    showTeacher(q,3,3);
    
    freeTeacher(&q,3,3);
    
    
    return 0;
}
```

#### 1.7、结构体数组排序

```c
typedef struct Teacher
{
    int age;
    char **stu;		//一个老师有多个学生
}Teacher;

void sortTeacher(Teacher *q,int n)
{
    if(q==NULL)
        return -1;
    int i=0;
    int j=0;
    Teacher tmp;
    for(i=0;i<n-1;i++)
    {
        for(j=i+1;j<n;j++)
        {
            if(p[i].age<p[j].age)
            {
                tmp=p[i];
                p[i]=p[j];
                p[j]=tmp;
            }
        }
    }
}
```

#### 1.8、结构体深拷贝和浅拷贝

1）浅拷贝

* 结构体中嵌套指针，而且动态分配空间
* 同类型结构体变量赋值
* 不同的结构体成员指针指向同一块内存

```c
typedef struct Teacher
{
    char *name;
    int age;
}Teacher;

int main()
{
    Teacher t1;
    t1.name=(char*)malloc(30);
    strcpy(t1.name,"lily");
    t1.age=22;
    
    Teacher t2;
    t2=t1;				//浅拷贝，t1.name 和 t2.name 都指向堆区的同一块内存
    printf("%s %d\n",t2.name,t2.age);		//lily 22
    
    //释放
    if(t1.name!=NULL)
    {
        free(t1.name);
        t1.name=NULL;
    }
    
    if(t2.name!=NULL)		//err，由于t1.name 和 t2.name 都指向堆区的同一块内存，前面已经是放过了，这里释放的是与前面同一块内存，同一块内存释放两次，就会报错
    {
        free(t2.name);
        t2.name=NULL;
    }
    
    return 0;
}
```

2）深拷贝

```c
typedef struct Teacher
{
    char *name;
    int age;
}Teacher;

int main()
{
    Teacher t1;
    t1.name=(char*)malloc(30);
    strcpy(t1.name,"lily");
    t1.age=22;
    
    Teacher t2;    
    //深拷贝，人为地分配内存
    t2.name=(char*)malloc(30);
    t2=t1;				//深拷贝，此时t1.name 和 t2.name 指向堆区的不同内存
    
    printf("%s %d\n",t2.name,t2.age);		//lily 22
    
    //释放
    if(t1.name!=NULL)
    {
        free(t1.name);
        t1.name=NULL;
    }
    
    if(t2.name!=NULL)		//OK
    {
        free(t2.name);
        t2.name=NULL;
    }
    
    return 0;
}
```

#### 1.9、结构体偏移量

* 结构体定义下来，内部成员变量的内存布居已经确定

```c
typedef struct Teacher
{
    char name[64];			//64
    int age;				//4
    int id;					//4
}Teacher;

int main()
{
    Teacher t1;
    Teacher *p=NULL;
    p=&t1;
    
    int n1=(int)(&p->age)-(int)p;		//64		相对于结构体首地址
    int n2=&((Teacher*)0)->age;			//决定 0 地址的偏移量
    printf("%d\n",n2);					//64
    
    
    return 0;
}
```

#### 1.10、结构体内存对齐

1）对齐规则

* 数据成员的对齐规则（以最大的类型的字节大小为单位）：结构体的数据成员，第一个数据成员放在偏移量为 0 的位置，以后每个数据成员放在偏移量为该数据成员大小的整数倍的地方
* 结构体作为成员的对其规则：如果一个结构体 B 里嵌套结构体 A，则结构体 A 应该从偏移量为 A 内部最大成员的整数倍的地方开始存储。例如，结构体 A 中有 int、char、double 等类型的成员，那么 A 应该从偏移量为 8 的整数倍的位置开始存储。结构体 A 站的内存为该结构体成员内部最大元素的整数值，不足补齐。
* 收尾工作：结构体的总大小，即 sizeof 计算的大小，是其内部最大成员的整数倍，不足的部分要补齐。
* char 占一个字节，char 型数组，不管定义的数组有多大，都是只有一个字节

```c
struct stru1
{
    int a;
    short b;
    double c;
}A;			//对齐单位为 8 个字节，a 放在 0-7 字节内，b 放在 8-15 字节内，放不满的补齐，c 放在 16-23 字节
// A 的大小为 24 字节，sizeof(A)=24

struct stru2
{
    int a;						//0-7
    short b;					//8-15
    double c;					//16-23
    struct stru1 B;				//B 内部的每个成员也都按照以 8 字节进行对齐存储，B.a:24-31, B.b:32-39, B.c:40-48
}C;		//对齐单位为 8 个字节
//结构体嵌套：以最大的成员的内存大小为单位进行对齐，
```

2）指定对齐单位

* 指定的对齐单位小于最大成员，则以指定的对齐单位进行对齐
* 指定的对齐单位大于最大成员，则以最大成员的大小为单位进行对齐

```c
#pragma pack(2)			//指定对其单位为 2 个字节
struct stru1
{
    int a;				//以两个单位来对齐
    short b;			//以一个单位来对齐
    double c;			//以四个单位来对齐
}A;	
```

3）不完整类型的字节对齐：位域

* 一个位域不许存储在同一个字节中，不可以跨字节。例如，一个字节所剩空间不够存另一位域时，应从下一单元存放该位域
* 如果相邻位域字段的类型相同，且其位宽之和小于类型的 sizeof 大小时，则后面的字段将紧邻前一个字段存储，直到不能容纳为止
* 如果相邻位域字段的类型相同，但其位宽之和大于类型的 sizeof 大小时，则后面的字段将从新的存储单元开始，其偏移量为其类型大小的整数倍
* 如果相邻位域字段的类型不相同，则各编译器的具体实现不同，VC6 采取不压缩的方式，Dev-C++ 和 gcc 采取压缩的方式
* 整个结构体的总大小为最宽基本类型成员大小的整数倍

```c
struct A
{
   	int a1:5;		//a1 指定位域宽度为 5 位
   	int a2:			//a2 指定位域宽度为 9 位
    char c;
    int b:4;
    short s;
}B;		//sizeof(b)=16
//a1+a2=14 位，少于32 位，占据32位，即 4 字节
//c，也需要 4 字节
// b，占 4 位，少于32 位，占据32位，即 4 字节
// s，也需要 4 字节
//共 16 字节

```
