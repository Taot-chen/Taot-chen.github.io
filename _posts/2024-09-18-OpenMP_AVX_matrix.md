---
layout: post
title: OpenMP_AVX_matrix
date: 2024-09-18
tags: [cpp]
author: taot
---

## OpenMP+AVX加速矩阵运算

### 1 调整计算顺序

一般的方阵乘法的实现：
```cpp
void MatrixMul(int **A, int **B, int **C, int size) {
    for (int i = 0; i < size; i++)
        for (int j = 0; j < size; j++)
            for (int k = 0; k < size; k++)
                C[i][j] += A[i][k] * B[k][j];
}
```
根据线性代数的知识容易知道，`C[i][j]` 的计算结果和 ijk 的循环顺序没有关系。仔细观察上面的代码可以发现，最内层循环的循环变量是 k，那么此时 A 是逐行访问的，而 B 则是逐列访问的。根据高维数组的存储方式不难知道，在这样的情况下，矩阵 A 是的访问在内存上是连续的，而 B 的访问在内存上是不连续的。

为了方便使用 AVX 并行计算，以及访问 cache，可以调整循环下标的顺序为 ikj，这样矩阵 A 和 B 的访问在内存上都是连续的了，考虑到二维数组和一维数组的内存布局是一样的门卫了方便，可以使用一维数组来处理：
```cpp
void conventional_MatMul(float *A, float *B, float *C, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            for (int k = 0; k < size; k++) {
                *(C + i * size + j) += *(A + i * size + k) * *(B + k * size + j);
            }
        }
    }
}
```

### 2 AVX 优化

对于最内存层循环，
```cpp
for (int j = 0; j < size; j++)
    *(C + i * size + j) += *(A + i * size + k) * *(B + k * size + j);
```
计算分为两步进行： *(A + i * size + k) 与 B 的第 k 行相乘；相乘的结果与 C 的第 i 行相加。

#### 2.1 乘法优化

考虑到最内层循环里面，`*(A + i * size + k)`是一个固定的数，与 B 的每一行中的每一个数相乘，因此存放 `*(A + i * size + k)` 的寄存器的每个单元都要写入相同的`*(A + i * size + k)`，可以使用 `_mm512_set1_ps`方法进行复制型的加载。
```cpp
__m512 ra = _mm512_set1_ps(*(A + i * size + k));
```
B 的加载就想对比较方便，现在 avx 的内存对齐和不对齐的性能差不多，就不做内存对齐，使用 _mm512_loadu_ps：
```cpp
_mm512_loadu_ps(B + k * size + j)；
```

乘法方面注意要使用 `_mm512_mul_ps` 来得到每一位相乘的结果。
```cpp
rb = _mm512_mul_ps(ra, rb);
```

#### 2.2 加法优化

加法和乘法类似，先加载矩阵 C：
```cpp
_mm512_loadu_ps(C + k * size + j)；
```

和乘法的结果相加。
```cpp
rc = _mm512_add_ps(rb, rc);
```

最后写回C，
```cpp
_mm512_storeu_ps(C + i * size + j, rc);
```


#### 2.3 循环步长

使用 AVX512， 一次可以处理 512 位，即 16 个 float32，因此循环步长设置为 16。
```cpp
#include <iostream>
#include <immintrin.h>
#include <cstdlib> 
#include <ctime>

void gen_matrix(float *m, int size) {
    srand (static_cast <unsigned> (time(0)));
    for (int i = 0; i < size; i++) {
        for (int j= 0; j < size; j++)
            *(m + i * size + j) = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
    }
}

void conventional_MatMul(float *A, float *B, float *C, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            for (int k = 0; k < size; k++) {
                *(C + i * size + j) += *(A + i * size + k) * *(B + k * size + j);
            }
        }
    }
}

void avx512_MatMul(float *A, float *B, float *C, int size) {
    for (int i = 0; i < size; i++) {
        for (int k = 0; k < size; k++) {
            int j = 0;
            __m512 ra = _mm512_set1_ps(*(A + i * size + k));
            for (; j <= size; j += 16) {
                 __m512 rc;
                rc = _mm512_add_ps(_mm512_mul_ps(ra, _mm512_loadu_ps(B + k * size + j)), rc);
                _mm512_storeu_ps(C + i * size + j, rc);
            }
        }
    }
}

void avx256_MatMul(float *A, float *B, float *C, int size)
{
    for (int i = 0; i < size; i++) {
        for (int k = 0; k < size; k++) {
            int j = 0;
            __m256 ra = _mm256_set1_ps(*(A + i * size + k));
            for (; j <= size; j += 8) {
                __m256 rc;
                rc = _mm256_add_ps(_mm256_mul_ps(ra, _mm256_loadu_ps(B + k * size + j)), rc);
                _mm256_storeu_ps(C + i * size + j, rc);
                
            }
        }
    }
}


int main() {
    int size = 2000;
    float* A = new float[size * size]();
    float* B = new float[size * size]();
    float* C1 = new float[size * size]();
    float* C2 = new float[size * size]();
    float* C3 = new float[size * size]();
    gen_matrix(A, size);
    gen_matrix(B, size);

    std::cout << "conventional MatMul: " << std::endl;
	clock_t begin_time1 = clock();
    conventional_MatMul(A, B, C1, size);
	float seconds1 = float(clock() - begin_time1) / CLOCKS_PER_SEC;
	std::cout << "time: " << seconds1 << "s\n" << std::endl;

    std::cout <<"avx256 MatMul: " << std::endl;
	clock_t begin_time2 = clock();
    avx256_MatMul(A, B, C2, size);
	float seconds2 = float(clock() - begin_time2) / CLOCKS_PER_SEC;
	std::cout << "time: " << seconds2 << "s\n" << std::endl;

    std::cout <<"avx512 MatMul: " << std::endl;
	clock_t begin_time3 = clock();
    avx512_MatMul(A, B, C3, size);
	float seconds3 = float(clock() - begin_time3) / CLOCKS_PER_SEC;
	std::cout << "time: " << seconds3 << "s\n" << std::endl;

    delete[] A;
    delete[] B;
    delete[] C1;
    delete[] C2;
    delete[] C3;

    return 0;
}
```


编译运行，
```bash
g++ mm_avx.cpp -march=native -o mm_avx
./mm_avx

# conventional MatMul:
# time: 41.7601s

# avx256 MatMul:
# time: 5.04586s

# avx512 MatMul:
# time: 3.0,171s
```

对比 2000*2000 的矩阵乘法，使用 avx256 的速度大概是常规的方式的 8 倍多，使用 avx512 的速度大概接近常规的方式的 13 倍。


### 3 OpenMP 优化

相较于 AVX，OpenMP 在使用方面就简单了许多，在这里只需要在最外层循环上放加上一句`#pragma omp parallel for`即可，在末尾加上`num_threads(N)`设置使用的线程数为 N。
```cpp
#include <iostream>
#include <immintrin.h>
#include <cstdlib> 
#include <ctime>
#include <omp.h>
#include <chrono>

void gen_matrix(float *m, int size) {
    srand (static_cast <unsigned> (time(0)));
    for (int i = 0; i < size; i++) {
        for (int j= 0; j < size; j++)
            *(m + i * size + j) = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
    }
}

void conventional_MatMul(float *A, float *B, float *C, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            for (int k = 0; k < size; k++) {
                *(C + i * size + j) += *(A + i * size + k) * *(B + k * size + j);
            }
        }
    }
}

void avx512_MatMul(float *A, float *B, float *C, int size, int thread_num) {
    #pragma omp parallel for num_threads(thread_num)
    for (int i = 0; i < size; i++) {
        for (int k = 0; k < size; k++) {
            int j = 0;
            __m512 ra = _mm512_set1_ps(*(A + i * size + k));
            for (; j <= size; j += 16) {
                 __m512 rc;
                rc = _mm512_add_ps(_mm512_mul_ps(ra, _mm512_loadu_ps(B + k * size + j)), rc);
                _mm512_storeu_ps(C + i * size + j, rc);
            }
        }
    }
}

void avx256_MatMul(float *A, float *B, float *C, int size, int thread_num) {
    #pragma omp parallel for num_threads(thread_num)
    for (int i = 0; i < size; i++) {
        for (int k = 0; k < size; k++) {
            int j = 0;
            __m256 ra = _mm256_set1_ps(*(A + i * size + k));
            for (; j <= size; j += 8) {
                __m256 rc;
                rc = _mm256_add_ps(_mm256_mul_ps(ra, _mm256_loadu_ps(B + k * size + j)), rc);
                _mm256_storeu_ps(C + i * size + j, rc);
                
            }
        }
    }
}


int main() {
    int size = 2000;
    int thread_num = 32;
    float* A = new float[size * size]();
    float* B = new float[size * size]();
    float* C1 = new float[size * size]();
    float* C2 = new float[size * size]();
    float* C3 = new float[size * size]();
    gen_matrix(A, size);
    gen_matrix(B, size);

    std::cout << "conventional MatMul: " << std::endl;
	auto start1 = std::chrono::steady_clock::now();
    conventional_MatMul(A, B, C1, size);
    auto end1 = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::micro> elapsed1 = end1 - start1;
	float seconds1 = float(elapsed1.count()) / 1000000;
	std::cout << "time: " << seconds1 << "s\n" << std::endl;

    std::cout <<"avx256 MatMul: " << std::endl;
	auto start2 = std::chrono::steady_clock::now();
    avx256_MatMul(A, B, C2, size, thread_num);
	auto end2 = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::micro> elapsed2 = end2 - start2;
	float seconds2 = float(elapsed2.count()) / 1000000;
	std::cout << "time: " << seconds2 << "s\n" << std::endl;

    std::cout <<"avx512 MatMul: " << std::endl;
	auto start3 = std::chrono::steady_clock::now();
    avx512_MatMul(A, B, C3, size, thread_num);
	auto end3 = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::micro> elapsed3 = end3 - start3;
	float seconds3 = float(elapsed3.count()) / 1000000;
	std::cout << "time: " << seconds3 << "s\n" << std::endl;

    delete[] A;
    delete[] B;
    delete[] C1;
    delete[] C2;
    delete[] C3;

    return 0;
}
```

编译运行，
```bash
g++ mm_avx.cpp -std=c++11 -march=native -g -pthread -Wno-format -fpermissive -fopenmp  -o mm_avx
./mm_avx

# conventional MatMul:
# time: 41.8979s

# avx256 MatMul:
# time: 0.302015s

# avx512 MatMul:
# time: 0.193028s
```

在开 16 线程的条件下，对比 2000*2000 的矩阵乘法，使用 avx256 的速度大概接近常规的方式的 140 倍，使用 avx512 的速度大概接近常规的方式的 220 倍。

初次接触 avx 和 openmp，使用还不熟悉，但是从结果来看，在矩阵乘上的加速效果非常可观。
