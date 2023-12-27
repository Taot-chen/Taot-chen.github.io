---
layout: post
title: c_basic_7_customized_types
date: 2023-12-27
---

## 复合类型（自定义类型）

### 1、结构体

#### 1.1、概述

* 将不同类型的数据组合成一个有机的整体。

#### 1.2、结构体变量的定义和初始化

1）结构体变量的定义

* 先声明结构体类型再定义变量名
* 在声明类型的同时定义变量
* 直接定义结构体类型变量（无类型名）

2）结构体类型和结构体变量的关系

* 结构体类型：相当于一个模型，但其中并无具体数据，系统对之也不分配实际内存单元
* 结构体变量：系统根据结构体类型（内部成员状况）为之分配空间

3）结构体的定义格式：

```c
struct 结构体名
{
	成员列表
}
```

#### 1.3、结构体成员的使用

```c
//结构体定义
struct student
{
    char name[21];
    int age;
    int score;
    char addr[51];
}stu1,stu2,stu3;		//在定义结构体的时候可以直接进行变量创建和赋值

int main()
{
    //创建结构体变量
    //结构体类型 结构体变量名
     /*
    struct student stu;
    
   
    //stu.name="张三";	err，字符数组不能被赋值
    strcpy(sstu.name,"张三");
    stu.age=18;
    stu.score=100;
    //stu.addr="北京市朝阳区";
    strcpy(stu.addr,"北京市朝阳区");
    */
    
    //另一种快速赋值的方式
    struct student stu=("张三",18,100,"北京市朝阳区");
    
    printf("姓名：%s\n",stu.name);
    printf("年龄：%d\n",stu.age);
    printf("成绩：%d\n",stu.score);
    printf("地址：%s\n",stu.addr);
    
    
    return 0;
}
```

1）从键盘输入初始化结构体

```c
//结构体定义
struct student
{
    char name[21];
    int age;
    int score;
    char addr[51];
};

int main()
{
    struct student stu;
    scanf("%s%d%d%s",stu.name,&stu.age,&stu.score,stu.addr);
    printf("姓名：%s\n",stu.name);
    printf("年龄：%d\n",stu.age);
    printf("成绩：%d\n",stu.score);
    printf("地址：%s\n",stu.addr);
    
    return 0;
}
```

#### 1.4、结构体数组

```c
//结构体定义
struct student
{
    char name[21];
    int age;
    char sex;
    int score[3];
    char addr[51];
};

int main()
{
    struct student stu[3]=
    {
        {"张三",22,'M',88,99,10,"河北唐山"},
        {"李四",23,'F',59,59,59."河北邯郸"},		//也可以这样：{.addr="河北邯郸","李四",23,'F',59,59,59},
        {"王五",18,'M',100,100,100,"黑龙江大庆"}
    };
    //结构体数组元素个数计算
    printf("结构体数组大小：%d\n",sizeof(stu));						//288
    printf("结构体数组元素大小：%d\n",sizeof(stu[0]));				//96
    printf("结构体数组大小：%d\n",sizeof(stu)/sizeof(stu[0]));			//3
    
    for(int i=0;i<3;i++)
    {
        printf("姓名：%s\n",stu[i].name);
        printf("年龄：%d\n",stu[i].age);
        printf("性别：%s\n",stu[i].sex='M'?"男":"女");
        printf("成绩1：%d\n",stu[i].score[0]);
        printf("成绩2：%d\n",stu[i].score[1]);
        printf("成绩3：%d\n",stu[i].score[2]);
        printf("地址：%s\n",stu[i].addr);
    }
    return 0;
}
```

1）结构体元素个数计算

```c
//结构体数组元素个数计算
    printf("结构体数组大小：%d\n",sizeof(stu));						//288
    printf("结构体数组元素大小：%d\n",sizeof(stu[0]));				//96
    printf("结构体数组大小：%d\n",sizeof(stu)/sizeof(stu[0]));			//3
```

2）结构体成员需要偏移对齐

* 上面的结构体数组元素大小原本是：21+4+1+3*4+51=89字节，实际输出是 96字节。
* 结构体需要偏移对齐，以最大的数据类型为基准。

#### 1.5、结构体数组排序

```c
int main()
{
	struct student stu[3]=
    {
        {"张三",22,'M',88,99,10,"河北唐山"},
        {"李四",23,'F',59,59,59."河北邯郸"},		
        {"王五",18,'M',100,100,100,"黑龙江大庆"}
    };
    
    //按照年龄排序
    for(int i=0;i<3-1;i++)
    {
        for(int j=0;j<3-1-i;j++)
        {
            if(stu[j].age>stu[j+1].age)
            {
                struct student temp=stu[j];
                stu[j]=stu[j+1];
                stu[j+1]=temp;
            }
        }
    }
    for(int i=0;i<3;i++)
    {
        printf("姓名：%s\n",stu[i].name);
        printf("年龄：%d\n",stu[i].age);
        printf("性别：%s\n",stu[i].sex='M'?"男":"女");
        printf("成绩1：%d\n",stu[i].score[0]);
        printf("成绩2：%d\n",stu[i].score[1]);
        printf("成绩3：%d\n",stu[i].score[2]);
        printf("地址：%s\n",stu[i].addr);
    }
    return 0;
}

```

#### 1.6、开辟堆空间存储结构体

```c
typedef struct student ss;

struct student
{
    char name[21];
    int age;
    char sex;
    int score[3];
    char addr[51];
};

int main()
{
    //printf("%d\n",sizeof(struct student));		//96
    //开辟堆区存储结构体数组
    //struct student* p=(struct student*)macllo(sizeof(struct student));
    ss* p=(ss*)macllo(sizeof(ss)*3);
    printf("结构体指针的大小：%d\n",sizeof(p));		//32位系统输出4,64位系统输出8
    
    //赋值
    for(int i=0;i<3;i++)
    {
        scanf("%s%d%,c%d%d%d%s",p[i].name,&p[i].age,&p[i].sex,		//这里加逗号是为了防止输入时候用于分隔的空格被读入
              &p[1].score[0],&p[1].score[1],&p[1].score[2],p[i].addr);
    }
    
    for(int i=0;i<3;i++)
    {
        printf("姓名：%s\n",p[i].name);
        printf("年龄：%d\n",p[i].age);
        printf("性别：%s\n",p[i].sex='M'?"男":"女");
        printf("成绩1：%d\n",p[i].score[0]);
        printf("成绩2：%d\n",p[i].score[1]);
        printf("成绩3：%d\n",p[i].score[2]);
        printf("地址：%s\n",p[i].addr);
    }
    //释放空间
    free(p);
    return 0;
}
```

#### 1.7、结构体嵌套结构体

```c
/*
struct 技能
{
	名称
	等级
	伤害
	范围
	耗蓝
	冷却
};

struct 人物信息
{
	等级
	经验
	金钱
	hp
	mp
	力量
	智力
	敏捷
	struct 技能 skills[4]
};

struct 人物信息 info;
info.skills[0].名称;
*/

struct scores
{
    int c;
    int cpp;
    int cs;
};

struct student
{
    char name[21];
    int age;
    struct scores ss;
    char addr[51];
};

int main()
{
    struct student stu={"貂蝉",18,99,99,99,"徐州"};
    printf("%s\n%d\n%d\n%d\n%d\n%s\n",stu.name,stu.age,stu.ss.c,stu.ss.cpp,stu.ss.cs,stu.a);
    
    printf("成绩结构体大小：%d\n",sizeof(struct scores));		//92
    printf("学生结构体大小：%d\n",sizeof(struct student));		//92
    return 0;
}
```

#### 1.8、结构体赋值

* 结构体赋值，会产生一个独立的变量（结构体成员为非指针的情况下）

```c
struct student
{
    char name[21];
    int age;
    int score;
    char addr[51];
};

int main()
{
    struct student stu={"孙尚香",26,60,"巴蜀"};
    struct student s1=stu;		//结构体赋值，会产生一个独立的变量
    strcpt(s1.name,"甘夫人");
    
    printf("%s\n",stu.name);	//孙尚香	这里s1和stu是两个独立的内存空间
}
```

#### 1.9、结构体和指针

1）指向普通结构体变量的指针

* 开辟堆空间存储结构体

2）结构体成员为指针类型

```c
struct student
{
    char* name;
    int age;
    int* scores;
    char* addr;
};

int main()
{
    struct student stu;
    //stu.name="张三";	如此赋值没错，但是name指向一个字符串常量，不可修改
    stu.name=(char*)malloc(sizeof(char)*21);
    strcpy(stu.name,"张三")；
    stu.scores=(int*)malloc(sizeof(int)*3);
    stu.addr=(char*)malloc(sizeof(char)*51);
    str.age=18;
    stu.scores[0]=88;
    stu.scores[1]=77;
    stu.scores[2]=99;
    strcpy(stu.addr,"北京市昌平区");
    
    free(stu.name);
    free(stu.scores);
    free(stu.addr);
    return 0;
}
```

3）结构体指针：指向结构体的指针

* 通过指针访问结构体成员，利用指针的取值运算

```c
//通过指针访问结构体成员，利用指针的取值运算
//结构体变量.成员
printf("%s\n",(*p).name);
printf("%d\n",(*p).age);
```

* 过指针访问结构体成员，利用指针的指向运算符

```c
//通过指针访问结构体成员，利用指针的指向运算符
//结构体指针->成员
printf("%s\n",p->name);
printf("%d\n",p->age);
    
```



```c
struct student
{
    char name[21];
    int age;
    int scores;
    char addr[51];
};

int main()
{
    struct student stu={"林冲",30,100,100,100,"汴京"};
    struct student* p;
    p=&stu;
    
    //通过指针访问结构体成员，利用指针的取值运算
    printf("%s\n",(*p).name);
    printf("%d\n",(*p).age);
    
    //通过指针访问结构体成员，利用指针的指向运算符
    printf("%s\n",p->name);
    printf("%d\n",p->age);
    
    return 0;
}
```

```c
typedef struct student ss;
struct student
{
	char* name;
    int age;
    int* scores;
    char* addr;
};

int main()
{
    ss* p=(ss*)malloc(sizeof(ss)*3);
    
    //开辟堆空间
    for(int i=0;i<3;i++)
    {
        (p+i)->name=(char*)malloc(sizeof(char)*21);
        p[i].scores=(int*)malloc(sizeof(int)*3);
        p[i].addr=(char*)malloc(sizeof(char)*51);
    }
    
    //输入
    for(int i=0;i<3;i++)
    {
        scanf("%s%d%d%d%d%s",(p+i)->name,&p[i].age,&p[i].scores[0],
              p[i].scores[1],&p[i].scores[2],p[i].addr);
    }
    
    //输出
    for(int i=0;i<3;i++)
    {
        printf();
    }
    
    //释放堆空间
    for(int i=0;i<3;i++)
    {
        free(p[i].name);
        free((p+i)->scores);
        free(p[i].addr);
    }
    free(p);
}
```

#### 1.10、结构体做函数参数

1）结构体普通变量做函数参数：值传递

```c
/*
typedef struct student ss;
struct student
{
    char name[21];
    int age;
    int score;
    char addr[51];
};

void func(ss stu)	//值传递
{
    strcpy(stu.name,"卢俊义");
    printf("%s\n",ss.name);		//卢俊义
}
int main()
{
    ss stu={"宋江",50,101,"水泊梁山"};
    func(stu);		//值传递
    printf("%s\n",stu.name);	//宋江
    
    return 0;
}
*/
typedef struct student ss;
struct student
{
    char* name;
    int age;
    int score;
    char addr[51];
};

void func(ss stu)	//值传递
{
    strcpy(stu.name,"卢俊义");
    printf("%s\n",ss.name);		//卢俊义
}

void func1(ss stu)	//值传递
{
    stu.name=(char*)malloc(sizeof(char)*21);
    strcpy(stu.name,"卢俊义");
    printf("%s\n",ss.name);		//卢俊义
}
int main()
{
    ss stu={NULL,50,101,"水泊梁山"};
    stu.name=(char*)malloc(sizeof(char)*21);
    strcpy(stu.name,"宋江");
    func1(stu);		//值传递
    printf("%s\n",stu.name);	//宋江
    
    func(stu);		//值传递
    printf("%s\n",stu.name);	//卢俊义
    
    return 0;
}

```

2）结构体指针变量做函数参数：地址传递

```c
typedef struct student ss;
struct student
{
    char name[21];
    int age;
    int score;
    char addr[51];
};

void func(ss* p)	//地址传递
{
    strcpy(p->name,"公孙胜");
    printf("%s\n",p->name);		//公孙胜
}
int main()
{
    ss stu={"吴用",50,101,"水泊梁山"};
    func(&stu);		//地址传递
    printf("%s\n",stu.name);	//公孙胜
    
    return 0;
}
```

3）结构体数组做函数参数：退化为指针，地址传递

```c
typedef struct student ss;
struct student
{
    char name[21];
    int age;
    int score;
    char addr[51];
};

//数组作为函数参数，退化为指针，需要传递元素个数
void BubbleSort(ss* stu,int length)	//地址传递
{
    printf("%d\n",sizeof(stu));
    for(int i=0;i<3-1;i++)
    {
        for(int j=0;j<3-i-1;j++)
        {
            if(stu[j].age>stu[j+1].age)
            {
                ss temp=stu[j];
                stu[j]=stu[j+1];
                stu[j+1]=temp;
            }
        }
    }
}

int main()
{
    ss stu[3]=
    {
        {"鲁智深",30,33,"五台山"},
        {"呼延灼",45,44,"汴京"},
        {"顾大嫂",28,55,"汴京"}
    };
    BubbleSort(stu,3);		//输出：4	即指针的大小
    
    
    return 0;
}
```

4）const 修饰结构体指针形参变量

* 与 const 修饰其他变量形参是一致的
* const 修饰结构体指针类型

```c
typedef struct student ss;
struct student
{
    char name[21];
    int age;
    int scores[3];
    char addr[51];
};

int main()
{
    ss stu1={"孙悟空",700,101,101,101,"花果山"};
    ss stu2={"猪八戒",1200,1001,1001,1001,"高老庄"};
    const ss* p=&stu1;		//const 修饰结构体指针类型
    p=&stu2;	//OK
    p->age=888;		//err
    (*p).age=888;	//err
    return 0;
}
```

* const 修饰结构体指针变量

```c
typedef struct student ss;
struct student
{
    char name[21];
    int age;
    int scores[3];
    char addr[51];
};

int main()
{
    ss stu1={"孙悟空",700,101,101,101,"花果山"};
    ss stu2={"猪八戒",1200,1001,1001,1001,"高老庄"};
    ss* const p=&stu1;		//const 修饰结构体指针变量
    p=&stu2;	//err
    p->age=888;		//OK
    (*p).age=888;	//OK
    strcpy(p->name,"沙悟净");		//OK
    return 0;
}
```

* const 修饰结构体指针变量和指针类型

```c
typedef struct student ss;
struct student
{
    char name[21];
    int age;
    int scores[3];
    char addr[51];
};

int main()
{
    ss stu1={"孙悟空",700,101,101,101,"花果山"};
    ss stu2={"猪八戒",1200,1001,1001,1001,"高老庄"};
    const ss* const p=&stu1;		//const 修饰结构体指针变量和指针类型
    p=&stu2;	//err
    p->age=888;		//err
    (*p).age=888;	//err
    strcpy(p->name,"沙悟净");		//err
    
    ss** pp=&p;
    (*pp)->age=1234;	//OK
    *pp=&stu2;			//OK
    
    **PP.age=789;		//OK
    
    return 0;
}
```

### 2、共用体（联合体）

* 联合体 union 还一个能在同一个存储空间存储不同类型数据的类型；
* 联合体所占的内存长度等于其最长成员的长度倍数；
* 同一内存段可以用来存放几种不同类型的成员，但是每一瞬时只有一种起作用；
* 共用体变量中起作用的成员是最后一次存放的成员，在存入一个新的成员后原有的成员的值会被覆盖；
* 共用体变量的地址和它的各成员的地址都是同一地址。

```c
union Var
{
    int a;
    float b;
    double c;
    char d;
    short f;
    //short f[6];	//如果这样，那么sizeof(Var)=16
    //内存对齐，最大类型的整数倍对齐
};

int main()
{
    union Var uVar;
    /*
    uVar.a=100;
    printf("%d\n",uVar.a);		//100
    */
    
    uVar.a=100;
    uVar.b=3.14;
    printf("%d\n",uVar.a);		//乱码
    printf("%f\n",uVar.b);		//3.14
    
    printf("%d\n",sizeof(uVar));		//8	大小和最大的类型有关
    
    printf("%p\n",&uVar);
    printf("%p\n",&uVar.a);
    printf("%p\n",&uVar.b);
    printf("%p\n",&uVar.c);
    printf("%p\n",&uVar.d);
    printf("%p\n",&uVar.e);
    printf("%p\n",&uVar.f);		//输出的地址相同
    
    return 0;
}
```

### 3、枚举

* 枚举：将变量值一一列举出来，变量的值只限于列举出来的值的范围内。

#### 3.1、枚举的定义：

```c
enum 枚举名
{
    枚举值表
}
```

* 在枚举值表中应列出所有可用值，也称为枚举元素
* 枚举值是常量，不能在程序中用赋值语句再对它赋值
* 枚举元素本身由系统定义了一个表示序号的数值从 0 开始顺序定义为：0、1、2、……

```c
enum color
{
    red,blue,green,pink,yellow,black,white
};

//也可以你使用系统默认的值，自定义赋值
enum color1
{
    red=10,blue,green,pink,yellow=20,black,white
};

int main()
{
    int value;
    scanf("%d",&value);
    enum color colorName;
    switch(value)
    {
    case red:
        printf("红色\n");
    	break;
    case blue:
        printf("蓝色\n");
    	break;
    case green:
        printf("绿色\n");
    	break;
    case pink:
        printf("粉色\n");
    	break;
    case yellow:
        printf("黄色\n");
    	break;
    case black:
        printf("黑色\n");
    	break;
    case white:
        printf("白色\n");
    	break;
    default:
        break;
    }
    return 0;
}
```

#### 3.2、枚举流程控制案例

```c
enum TYPE
{
    run,attack,skill,dance=10,showUI,frozen=20,dizzyn,dath,moti=30
    //值分别为：0、1、2、10、11、20、21、22、30
}type;

int main()
{
    int value;
    while(1)
    {
        scanf("%d",&value);
        switch(type)
    	{
        	case run:
                printf("正在移动中……\n");
                //value=30;
           	 	break;
        	case attack:
                printf("正在攻击中……\n");
            	break;
        	case skill:
                printf("正在施法中……\n");
            	break;
        	case dance:
                printf("正在跳舞中……\n");
            	break;
        	case showUI:
                printf("正在显示徽章中……\n");
            	break;
        	case frozen:
                printf("正在被冰冻中……\n");
            	break;
        	case dizzyn:
                printf("正在被眩晕中……\n");
            	break;
        	case dath:
                printf("死亡……\n");
                return 0;
            	break;
        	case moti:
                printf("等待释放命令中……\n");
            	break;
        	default:
            	break;
    	}
    }
    
    return 0;
}
```

### 4、typedefine

* 为一种数据类型（基本类型或自定义类型）定义一个新名字，不能创建新类型
* 与 #define 不同，typedef 仅限于数据类型，不能是表达式或者具体的值
* #define 发生在预处理，typedef 发生在编译阶段

1）为已经存在的数据类型取别名

```c
typedef unsigned int ui;
typedef struct student
{
    char name[21];
    int age;
    int score;
    char addr[51];
}ss;

int main()
{
    ui a=10;
    ss stu;
    return 0;
}
```

2）定义函数指针