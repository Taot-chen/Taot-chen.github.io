---
layout: post
title: SSE_AVX_basic
date: 2024-08-26
tags: [cpp]
author: taot
---

## SSE和AVX指令基本使用

SSE/AVX 指令属于 Intrinsics 函数，由编译器在编译时直接在调用处插入代码，避免了函数调用的额外开销。但是与 inline 函数不同，Intrinsics 函数的代码由编译器提供，能够更高效地使用机器指令进行优化调整。一般的函数是在库中，Intrinsic Function 内嵌在编译器中（built in to the compiler）。

优化器（Optimizer）内置的一些 Intrinsic Function 行为信息，可以对 Intrinsic 进行一些不适用于内联汇编的优化，所以通常来说 Intrinsic Function 要比等效的内联汇编（inline assembly）代码更快。优化器能够根据不同的上下文环境对 Intrinsic Function 进行调整。例如，以不同的指令展开 Intrinsic Function，将 buffer 存放在合适的寄存器中。

关于 SSE 和 AVX 内部函数的相关信息可以在[Intel® Intrinsics Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html)查看。


### 1 包含头文件

SSE 和 AVX 指令集有多个不同的版本， 其对应的 Intrinsic 包含在不同的头文件中，如果确定只使用某个版本的 SSE/AVX 指令则只包含相应的头文件即可。

如果不关心具体版本则可以使用`<intrin.h>`包含所有版本的头文件内容。

```cpp
#include <intrin.h> 
```

Intrinsic 头文件和 SIMD(SSE/AVX) 指令集，已经 Visual Studio 的版本对应关系

|Instrinsic 头文件|指令集描述|Visual Studio版本号|Visual Studio版本名|
|---|---|---|---|
|intrin.h|All Architectures|8.0|2005|
|mmintrin.h|MMX intrinsics|6.0|6.0 SP5+PP5|
|xmmintrin.h|Streaming SIMD Extensions intrinsics|6.0|6.0 SP5+PP5|
|emmintrin.h|Willamette New Instruction intrinsics (SSE2)|6.0|6.0 SP5+PP5|
|pmmintrin.h|SSE3 intrinsics|9.0|2008|
|tmmintrin.h|SSSE3 intrinsics|9.0|2008|
|smmintrin.h|SSE4.1 intrinsics|9.0|2008|
|nmmintrin.h|SSE4.2 intrinsics|9.0|2008|
|wmmintrin.h|AES and PCLMULQDQ intrinsics|10.0|2010|
|immintrin.h|Intel-specific intrinsics(AVX)|10.0|2010 SP1|
|ammintrin.h|AMD-specific intrinsics (FMA4, LWP, XOP)|10.0|2010 SP1|
|mm3dnow.h|AMD 3DNow! intrinsics|6.0|6.0 SP5+PP5|

在[Intel® Intrinsics Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html)也可以查询到每个函数所属的指令集和对应的头文件信息。

### 2 编译选项

使用 SSE/AVX 指令，除了包含头文件以外，还需要添加额外的编译选项，才能保证代码编译成功。各版本的 SSE 和 AVX 都有单独的编译选项，例如`-msseN`, `-mavxN`(N 表示版本编号)。此类编译选项支持向下兼容，比如`-msse4`可以编译 SSE2 的函数，`-mavx`也可以兼容各版本的SSE。

### 3 数据类型

Intel目前主要的SIMD指令集有 MMX, SSE, AVX, AVX-512，其对处理的数据位宽分别是：

* MMX 64位 
* SSE 128位 
* AVX 256位 
* AVX-512 512位 

每种位宽对应一个数据类型，数据类型名称包括三个部分：

* 前缀 `__m`，两个下划线加 m。
* 中间是数据位宽。
* 最后加上的字母表示数据类型，i 为整数，d 为双精度浮点数，不加字母则是单精度浮点数。

那么，对于 SSE 指令集具有的数据类型：`__m128`, `__m128i`, `__m128d`；
AVX 指令集则包括`__m256`, `__m256i`, `__m256d`;
`__m64` 对应 64 位对应的数据类型，该类型仅能供 MMX 指令集使用。由于 MMX 指令集也能使用 SSE 指令集的 128 位寄存器，故该数据类型使用的情况较少。

这里的位宽指的是 SIMD 寄存器的位宽，CPU 需要先将数据加载进专门的寄存器之后再并行计算。


### 4 Instrinsic 函数命名

Intrinsic 函数的命名通常由3个部分构成：

* 第一部分为前缀`_mm`，MMX 和 SSE 都为`_mm`开头，AVX 和 AVX-512 则会额外加上 256 和 512 的位宽标识。
* 第二部分表示执行的操作，比如`_add`, `_mul`, `_load`等，操作本身也会有一些修饰，比如`_loadu`表示以无需内存对齐的方式加载数据。
* 第三部分为操作选择的数据范围和数据类型，比如`_ps`的 p(packed) 表示所有数据，s(single) 表示单精度浮点; `_ss`则表示 s(single) 第一个，s(single) 单精度浮点; `_epixx`（xx 为位宽）操作所有的 xx 位的有符号整数，`_epuxx`则是操作所有的 xx 位的无符号整数。

例如`_mm256_load_ps`表示将浮点数加载进整个 256 位寄存器中。

SSE 指令集对分支处理能力比较差，而且从 128 位的数据中提取某些元素数据的代价又比较大，因此不适合有复杂逻辑的运算。

绝大部分 Intrinsic 函数都是按照这样的格式构成，每个函数都能在[Intel® Intrinsics Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html)找到更为完整的描述。


### 5 SSE 指令基本使用

考虑这样的场景，需要对A、B数组求和，并将结果写入数组C。

常规的 C 写法：
```cpp
#include <stdio.h>

#define SIZE 100000
int main()
{
    float A[SIZE], B[SIZE], C[SIZE];
    
    for(int i = 0; i < SIZE; i++)
        C[i] = A[i] + B[i];
}
```

使用 SSE 指令来优化，进行数据并行计算：
```cpp
// 导入头文件
#include <intrin.h>

// 创建 3 个 __m128 寄存器分别存储三个数组的 float 值
__m128 ra, rb, rc;

// 到了循环体内，我们需要把 A、B 的值写入寄存器之中，这就需要用到 _mm_loadu_ps 函数，他会把从指针位置开始的后 128 位的数据写入寄存器。
// 这里使用 loadu 而不是用 load 函数，和内存对齐有关，后面再作说明
ra = _mm_loadu_ps(A + i);   // A+i 等价于 &A[i]。
rb = _mm_loadu_ps(B + i);

// 用 _mm_add_ps 函数计算 ra、rb 相加，然后把结果返回到 rc 之中。
rc = _mm_add_ps(ra, rb);

// rc 的值还得写回到C数组，这要用到 _mm_storeu_ps 函数。
// 同样是和内存对齐有关，这里使用 storeu，不使用 store
_mm_storeu_ps(C + i, rc);

// 因为128位寄存器一次可以写入 4(128/32) 个 float 值，等于一次循环计算 4 个 float 的加法，循环的跨步长也应该由 1 变为 4，这样循环次数就只需要原来的 1/4。
for (int i = 0; i < SIZE; i += 4)
```

完整的代码如下，代码编译之后即可运行，编译的时候需要加编译选项：
```cpp
#include <stdio.h>
#include <intrin.h>

#define SIZE 100000
int main()
{
    float A[SIZE], B[SIZE], C[SIZE];

    for (int i = 0; i < SIZE; i += 4)    // 一次计算4个数据，所以要改成+4
    {
        __m128 ra = _mm_loadu_ps(A + i); // ra = {A[i], A[i+1], A[i+2], A[i+3]}
        __m128 rb = _mm_loadu_ps(B + i); // rb = {B[i], B[i+1], B[i+2], B[i+3]}
        __m128 rc = _mm_add_ps(ra, rb);  // rc = ra + rb
        _mm_storeu_ps(C + i, rc);        // C[i~i+3] <= rc
    }
}
```

这里的 SSE 的版本代码相比于前面纯 C 的版本，大概会有 1.7 倍左右的速度提升。

这里循环规模缩减为了纯 C 版本的 1/4，但是速度只提高了 1.7 倍，提升并没有循环缩减的规模大，这是因为上面的 SSE 版本代码里面用了非常多不必要的中间变量（ra、rb、rc都是），可以再做一些精简优化：
```cpp
#include <stdio.h>
#include <intrin.h>

#define SIZE 100000
int main()
{
    float A[SIZE], B[SIZE], C[SIZE];

    for (int i = 0; i < SIZE; i += 4)
    {
        _mm_storeu_ps(C + i,  _mm_add_ps(_mm_loadu_ps(A + i), _mm_loadu_ps(B + i)));
    }
}
```
此时的代码速度提升有差不多 2.2 倍了，这距离 4 倍的循环规模缩减还有一段距离，还可以通过内存对齐，类型转换和 AVX 来进一步提高速度。

### 6 内存对齐

#### 6.1 关于 loadu 和 load

loadu 表示无需内存对齐，不加 u 的版本需要原数据有 16 字节内存对齐，否则在读取的时候就会触发边界保护产生异常。

xx 字节对齐的意思是要求数据的地址是 xx 字节的整数倍，128 位宽的 SSE 要求 16 字节内存对齐，而 256 位宽的 AVX 函数则是要求32 字节内存对齐。

可以明显地看出，内存对齐要求的字节数就是指令需要处理的字节数，而要求内存对齐也是为了能够一次访问就完整地读到数据，从而提升效率。


#### 6.2 内存对齐

创建变量时设置 N 字节对齐可以用：

* `__declspec(align(N))`，MSVC 专用关键字
* `__attribute__((__aligned__(N)))`，GCC 专用关键字
* `alignas(N)`，C++11 关键字
只需要在创建变量时在类型名前加上这几个关键字，就像下面这样：
```cpp
alignas(16)                      float A[SIZE]; // C++11
__declspec(align(16))            float B[SIZE]; // MSVC
__attribute__((__aligned__(16))) float C[SIZE]; // GCC
```

对于 new 或 malloc 这种申请的内存也有相应的设置方法：

* `_aligned_malloc(size, N)`，包含在`<stdlib.h>`头文件中，与 malloc 相比多了一个参数 N 用于指定内存对齐。注意！用此方法申请的内存需要用`_aligned_free()`进行释放。
* `new((std::align_val_t) N)`，C++17 新特性，需要在 GCC7 及以上版本使用`-std=c++17`编译选项开启。

具体使用方式如下：
```cpp
float *A = new ((std::align_val_t)32) float[SIZE];             // C++17
float *B = (float *)_aligned_malloc(sizeof(float) * SIZE, 32); // <stdlib.h>
_aligned_free(B);                                              // 用于释放_aligned_malloc申请的内存
```

使用关键字把数组进行 16 字节内存对齐后，就可以把 loadu 和 storeu 替换成 load 和 store。
```cpp
#include <stdio.h>
#include <intrin.h>

#define SIZE 100000
int main()
{
    __attribute__((__aligned__(16))) float A[SIZE], B[SIZE], C[SIZE]; // GCC的内存对齐

    for (int i = 0; i < SIZE; i += 4)
    {
        _mm_store_ps(C + i,  _mm_add_ps(_mm_load_ps(A + i), _mm_load_ps(B + i))); // 用store和load替换storeu和loadu
    }
}
```

### 7 类型转换

`_mm_load_ps`函数和`_mm_store_ps`函数内部实际上只是对传入的指针进行了一次类型转换
```cpp
/* Load four SPFP values from P.  The address must be 16-byte aligned.  */
extern __inline __m128 __attribute__((__gnu_inline__, __always_inline__, __artificial__))
_mm_load_ps (float const *__P)
{
  return *(__m128 *)__P;
}

/* Store four SPFP values.  The address must be 16-byte aligned.  */
extern __inline void __attribute__((__gnu_inline__, __always_inline__, __artificial__))
_mm_store_ps (float *__P, __m128 __A)
{
  *(__m128 *)__P = __A;
}
```

这里的类型转换`*(__m128 *)__P`可以分成两部分来看：
* `(__m128 *)__P`：将`__P`从`float *`类型转换为`__m128 *`
* `*`：访问`__m128 *`指针指向的`__m128`对象

这里我们直接在代码中实现这一步，就可以去掉这两个函数调用的成本：
```cpp
#include <stdio.h>
#include <intrin.h>

#define SIZE 100000
int main()
{
    __attribute__((__aligned__(16))) float A[SIZE], B[SIZE], C[SIZE];

    for (int i = 0; i < SIZE; i += 4)
    {
        *(__m128 *)(C + i) = _mm_add_ps(*(__m128 *)(A + i), *(__m128 *)(B + i)); // 使用类型转换
    }
}
```
*注：转换成`__m128*`同样是有内存对齐要求的，若是低于 16 字节对齐就会在访问指针时出错，非对齐的情况应该使用`__m128_u*`指针。*

在前面的基础上，使用内存对齐和类型转换之后，相比于纯 C 的代码，速度提高到了 3.8 倍，这已经很接近循环缩减的 4 倍的期望值了。


### 8 AVX 

AVX 的用法与 SSE 相同，只需要根据命名规律修改一下数据类型和函数的名称就可以了。AVX 的数据处理位宽为 256 位，内存对齐要求对齐到 32 字节。
```cpp
#include <stdio.h>
#include <intrin.h>

#define SIZE 100000
int main()
{
    __attribute__((__aligned__(32))) float A[SIZE], B[SIZE], C[SIZE]; // 32字节对齐

    for (int i = 0; i < SIZE; i += 8) // 循环跨度修改为8
    {
        *(__m256 *)(C + i) = _mm256_add_ps(*(__m256 *)(A + i), *(__m256 *)(B + i)); // 使用256位宽的数据与函数
    }
}
```
使用 AVX，加上内存对齐和类型转换之后，相比于纯 C 的代码，速度提高到了 7 倍。

### 9 整数操作

SSE/AVX 整数计算使用的数据类型是`__m128i/__m256i`，二者都是以i结尾。

还有基本的算数函数，比如 SSE 的加法`add`，`epi`表示整数，后面的数字就是单个整数的数据位宽。比如`epi8`就是 1 字节 char 加法， 4 字节 int 加法就是`epi32`。

```cpp
__m128i _mm_add_epi8 (__m128i a, __m128i b)
__m128i _mm_add_epi16 (__m128i a, __m128i b)
__m128i _mm_add_epi32 (__m128i a, __m128i b)
__m128i _mm_add_epi64 (__m128i a, __m128i b)
```

整数乘法，以 SSE 为例：
```cpp
// mul
__m128i _mm_mul_epi32 (__m128i a, __m128i b)
__m128i _mm_mul_epu32 (__m128i a, __m128i b)
// mullo
__m128i _mm_mullo_epi16 (__m128i a, __m128i b)
__m128i _mm_mullo_epi32 (__m128i a, __m128i b)
__m128i _mm_mullo_epi64 (__m128i a, __m128i b)
// mulhi
__m128i _mm_mulhi_epi16 (__m128i a, __m128i b)
__m128i _mm_mulhi_epu16 (__m128i a, __m128i b)
```
有三种不同功能的乘法，建议使用之前到[Intel® Intrinsics Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html)查看一下功能描述。


