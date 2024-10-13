---
layout: post
title: openMP_multithread
date: 2024-08-26
tags: [cpp]
author: taot
---

## OpenMP多线程使用

OpenMP 是一套 C++ 并行编程框架, 也支持 Forthan，是一个跨平台的多线程实现， 能够使串行代码经过最小的改动自动转化成并行的。具有广泛的适应性。这个最小的改动，有时候只是一行编译原语！具体实现是通过分析编译原语#pragma，将用原语定义的代码块，自动转化成并行的线程去执行。每个线程都将分配一个独立的id. 最后再合并线程结果。

* 可以并行的条件：
  * 可拆分，代码和变量的前后不能相互依赖。
  * 独立运行，运行时，拥有一定的独有的资源。像独有的线程id等。


### 0 环境要求

#### 0.1 Windows / Visual Studio 平台

VS 版本不低于2015，都支持 OpenMP . 需要在 IDE 进行设置，才能打开 OpenMP 支持。

* 设置方式：调试 --> C/C++ --> 语言 --> OpenMP支持

这实际上使用了编译选项/openmp。 


#### 0.1 Linux / GCC 平台

Ubuntu 自带的 GCC 5.0.4, 直接支持选项 -fopenmp。后面都使用 ubuntu 系统来实验, windows 在配置好环境之后也可以使用。

### 1 验证支持 OpenMP

`main.cpp`
```cpp
#include <iostream>
using namespace std;

int main() {

    #if _OPENMP
        cout << " support openmp " << endl;
    #else
        cout << " not support openmp" << endl;
   #endif
   return 0;
}
```
编译并运行：
```bash
# 编译
g++ -std=c++11  -g -pthread -Wno-format -fpermissive -fopenmp -o main.o -c main.cpp
g++ -std=c++11  -g -pthread -Wno-format -fpermissive -fopenmp -o main main.o

# 运行
./main

# 输出
support openmp
```

也可以通过编写 Makefile 来更加优雅地编译代码。Makefile 文件的编写比较简单，这里给出一个示例：
`Makefile`
```cpp
CC = g++ -std=c++11 

nullstring := 
dep_I_incpath = -I./include/hiredis -I./include
dep_L_libpath = -L./lib
dep_l_libname = -lhiredis
#deps_extra = $(dep_I_incpath) $(dep_L_libpath)  $(dep_l_libname)

deps_extra = $(nullstring)

#c_flag = -g -Wno-format -fpermissive 
c_flag = -g -pthread -Wno-format -fpermissive -fopenmp

c_files =  \
$(wildcard *.cpp) $(wildcard *.c) $(wildcard *.cc)
h_files = \
$(wildcard *.hpp) $(wildcard *.h)
o_files_mess = \
$(patsubst %.cpp,%.o,${c_files}) $(patsubst %.c,%.o,${c_files}) $(patsubst %.cc,%.o,${c_files})
o_files = \
$(filter %.o, $(o_files_mess))
elf_file=  \
main

rm_files = \
*.o *.dep *.elf  *.s *.exe


all:$(elf_file)

$(elf_file):$(o_files)
	$(CC) $(c_flag) -o $@ $(o_files) $(deps_extra) 
main.o: main.cpp
	$(CC) $(c_flag) -o $@ -c $< $(deps_extra) 
#	@echo "c_file is:$(c_files)"; echo "h_files:$(h_files)"; echo "o_files:$(o_files)"

clean:
	rm -rf $(rm_files)

run:$(elf_file)
	./$(elf_file)
```

### 2 并行输出

* 顺序打印 0-10 可以使用 for 循环简单实现：
```cpp
for(int i=0; i< 10; i++) {
  cout << i << endl; 
}
```

* 用 OpenMP 实现并行输出，只需要在 for 循环的起始位置添加一行编译原语来标记即可：
```cpp
#pragma omp parallel for num_threads(4)
for(int i=0; i<10; i++) {
  	cout << i << endl; 
}
```

`main.cpp`
```cpp
#include <iostream>
#include <omp.h>

using namespace std; 

int serial_cout() {
	for(int i=0; i<10; i++) {
    	cout << i << endl; 
	}
	return 0; 
}



int parallel_cout() {

	#pragma omp parallel for num_threads(4)
	for(int i=0; i<10; i++) {
    	cout << i << endl; 
	}
	return 0; 
}


int main() {
    cout << "serial cout: " << endl;
    serial_cout();
    cout <<"parallel cout: " << endl;
    parallel_cout();
    return 0;
}
```

编译运行
```bash
g++ -std=c++11  -g -pthread -Wno-format -fpermissive -fopenmp -o main.o -c main.cpp
g++ -std=c++11  -g -pthread -Wno-format -fpermissive -fopenmp -o main main.o
./main

serial cout:
0
1
2
3
4
5
6
7
8
9
parallel cout:
0
1
2
36
7

4
5
8
9
```
可以看到串行输出的是从 0 到 9 顺序输出，并行输出是乱序的，并且并行输出的每次顺序都不一样，说明 OpenMP 并行起了作用。 

#### 2.1 说明

上面并行的代码新增了两行：
* `#include <omp.h>`，OpenMP 的编译头文件，包括一些常用API，像获取当前的线程id.

* `#pragma omp parallel for num_threads(4)`，用编译原语，指定其下面的代码块将会被渲染成多线程的代码，然后再编译。这里使用的线程数为 4 


### 3 三种 private

多个线程同时运行的情况下，变量的管理默认有下面两种情况：
* 创建线程之前就存在的变量——所有线程共享
* 线程运行时创建的局部变量——每个线程独有

#### 3.1 private

考虑这样的例子：
```cpp
#include <stdio.h>

int main()
{
    int i, j;

    #pragma omp parallel for
    for (i = 0; i < 3; i++)
        for (j = 0; j < 3; j++)
            printf("i=%d j=%d\n", i, j);
}
```
编译运行之后，会发现输出结果并不符合预期：
```bash
i=1 j=0
i=1 j=1
i=1 j=2
i=0 j=0
i=2 j=0
```
这是多个线程同时访问和修改相同变量导致的问题，

在 openmp 中，可以使用 `private()` 命令来那解决，`private(value)`，能让 每个线程都有该变量的独立拷贝，相当于一个同名的局部变量。
```cpp
#include <stdio.h>

int main() {
    int i, j;
    #pragma omp parallel for private(i, j)
    for (i = 0; i < 3; i++)
        for (j = 0; j < 3; j++)
            printf("i=%d j=%d\n", i, j);
}
```


#### 3.2 firstprivate

虽然private声明变量后，每个线程都会生成一个相应的拷贝。但这些线程并不会对他们进行初始化。考虑下面的情况：
```cpp
#include <stdio.h>

int main() {
    int x = -1;
    #pragma omp parallel for private(x)
    for (int i = 0; i < 5; i++)
        printf("x=%d\n", x);
}
```
可以看到拷贝出的变量值是随机的：
```bash
x=14908664
x=13903992
x=13909720
x=13905304
x=8
```

想要有初始化的话就需要用到 `firstprivate()`。
```cpp
#include <stdio.h>

int main() {
    int x = -1;
    #pragma omp parallel for firstprivate(x)
    for (int i = 0; i < 5; i++)
        printf("x=%d\n", x);
}
```

这样每个变量就都能初始化了
```bash
x=-1
x=-1
x=-1
x=-1
x=-1
```

#### 3.3 lastprivate

考虑这样的例子：
```cpp
#include <stdio.h>

int main() {
    int x = -1;
    printf("start x=%d\n", x);
    #pragma omp parallel for private(x)
    for (int i = 0; i < 5; i++)
        printf("x=%d\n", x = i);

    printf("final x=%d", x);
}
```

虽然现在我们有了初始化，但循环结束后变量的值还是无法保留:
```bash
start x=-1
x=1
x=2
x=3
x=0
x=4
final x=-1
```
前面使用private时，循环结束后x的值是不会保留的。

` lastprivate()` 的功能就是在 private 的基础上，能够在循环结束时保留变量的值。
```cpp
#include <stdio.h>

int main() {
    int x = -1;
    printf("start x=%d\n", x);
    #pragma omp parallel for lastprivate(x)
    for (int i = 0; i < 5; i++)
        printf("x=%d\n", x = i);

    printf("final x=%d", x);
}
```
输出结果：
```bash
start x=-1
x=3
x=2
x=1
x=4
x=0
final x=4
```
注意到，最后一个输出的是3，保存的结果却是4？这是因为 lastprivate 保存的变量是逻辑上的最后的值。从代码运行逻辑上来讲 x 最后的值是 4，所以结果就是 4。

#### 3.4 组合使用

从功能上可以看出，firstprivate 是对功能 private 的扩展，二者是相互替代的关系，所以 不能同时使用（会编译失败）；lastprivate 和 private 同样也是相互替代的关系，依旧 不能同时使用（会编译失败）；firstprivate 和 lastprivate 在功能上有相互补充关系，可以同时使用。


### 4 reduction

求1~10000000 之和, 用 16 个线程并行：
```cpp
#include <iostream>
#include <omp.h>
#include <ctime>

using namespace std; 

int parallel_sum() {

#if 0
// TODO : this is incorrect way to reduce !!!
	long long int sum = 0;
	#pragma omp parallel for num_threads(16)
	for(int i=0; i<10000000; i++) {
		sum +=  i; 
	}
	cout << sum << endl; 
#endif 

#if 1 

	long long int sum = 0;
	#pragma omp parallel for num_threads(16) reduction(+:sum)
	for(int i=0; i<10000000; i++) {
		sum +=  i; 
	}
	cout << sum << endl; 
#endif 
	return 0; 
}

int serial_sum() {
    long long int sum = 0;
    for(int i=0; i<10000000; i++) {
		sum +=  i; 
	}
	cout << sum << endl; 
    return 0;
}

int main() {
    cout << "serial sum: " << endl;
	clock_t begin_time1 = clock();    
    serial_sum();
	float seconds1 = float(clock( ) - begin_time1) / CLOCKS_PER_SEC;
	cout << "time: " << seconds1 << "s" << endl;
    cout <<"parallel sum: " << endl;
	clock_t begin_time2 = clock();
    parallel_sum();
	float seconds2 = float(clock( ) - begin_time2) / CLOCKS_PER_SEC;
	cout << "time: " << seconds2 << "s" << endl;
    return 0;
}
```

编译运行
```bash
g++ -std=c++11  -g -pthread -Wno-format -fpermissive -fopenmp -o main.o -c main.cpp
g++ -std=c++11  -g -pthread -Wno-format -fpermissive -fopenmp -o main main.o
./main

serial sum:
49999995000000
time: 0.046843s
parallel sum:
49999995000000
time: 0.004178s
```

不难看出，这里使用 opemmp 并行的时候，计算耗时更少。

#### 4.1 说明

在使用 openmp 并行求和的函数里面，在使用条件编译关掉的那个代码块里面，`sum += i` 这一行多个线程同时访问 sum 时产生会冲突导致。在这种情况下没法使用 private 对其进行求和，需要使用 reduction() 来解决，具体使用如 openmp 并行求和的函数里面，在使用条件编译打开的那个代码块所示。

reduction 的命令格式是 `reduction(operation : variable)`，其中 operation 是操作类型，variable 则是操作变量。

reduction的作用就是给每个线程创建一个独立的变量，在结束后根据操作类型进行归约。

* 默认操作包括:
  * 算数运算：+, *, -, max, min
  * 逻辑运算：&&, ||
  * 位运算：&, |, ^

这里我们需要对 sum 进行累加，所以应该使用 `reduction(+ : sum)`

### 5 获取线程 id

```cpp
#define DEFINE_idx auto idx = omp_get_thread_num();
#define _ROWS (omp_get_num_threads())
#pragma omp parallel for num_threads(3) 
for(int i=0; i<10; i++) {
    DEFINE_idx;
    printf("- idx is %d, i is %d, total thread num is %d\n", idx, i, _ROWS); 
}

// 输出
 idx is 0, i is 0, total thread num is 3
- idx is 0, i is 1, total thread num is 3
- idx is 0, i is 2, total thread num is 3
- idx is 0, i is 3, total thread num is 3
- idx is 2, i is 7, total thread num is 3
- idx is 2, i is 8, total thread num is 3
- idx is 2, i is 9, total thread num is 3
- idx is 1, i is 4, total thread num is 3
- idx is 1, i is 5, total thread num is 3
- idx is 1, i is 6, total thread num is 3
```

### 6 对于首个for循环迭代器的限制

OpenMP 在优化多重 for 语句时，会自动把第一个循环的迭代器（也就是i）设为 private。

比如在前面 private 里面的示例双重for循环中，如果将
```cpp
#pragma omp parallel for private(i, j)
```
修改为
```cpp
#pragma omp parallel for private(j) // 不显式声明i为private
```
修改后并不会影响循环的正常运行。
