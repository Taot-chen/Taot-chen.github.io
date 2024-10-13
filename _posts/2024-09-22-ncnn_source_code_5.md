---
layout: post
title: ncnn_source_code_5
date: 2024-09-22
tags: [ncnn]
author: taot
---

## NCNN 源码(2)-经典算子实现

### 1 Relu 算子

relu 是一个无参算子。ncnn 源码中，relu 算子是在`relu.h`和`relu.cpp`这两个文件里面。

由于 relu 比较简单，而 leaky relu 跟 relu 相比只是在小于 0 的地方乘了一个系数，所以作者就将 relu 和 leaky relu 写到一块，由一个只作用于小于 0 区域的斜率 slope 控制。

relu 算子算上初始化一共有6个方法：
* `ReLU`，声明了该 layer 是`one_blob_only`的，也就是单输入单输出的算子
* `load_param(FILE* paramfp)`，从文件指针加载参数
* `load_param_bin(FILE* paramfp)`，从 16 进制文件加载参数
* `load_param(const unsigned char*& mem)`，从数组加载参数
* `forward`，layer 推理
* `forward_inplace`，inplace 的 layer 推理

#### 1.1 load_param

ncnn 的模型由网络结构 param 文件和网络参数 bin 文件组成，由于 relu 的参比较简单，就是一个 float 数`slope`，所这个参是放到了 param 文件 layer 特定参数的位置，如下所示：
```bash
ReLU             relu_conv1       1 1 conv1 conv1_relu_conv1 0.000000
```
上面是从 param 中截取出来的一段，可以看到它后面的`slope`是 0，说明他是个 relu。

`ReLU::load_param(FILE* paramfp)`源码：
```cpp
int ReLU::load_param(FILE* paramfp) {
    int nscan = fscanf(paramfp, "%f", &slope);
    if (nscan != 1) {
        fprintf(stderr, "ReLU load_param failed %d\n", nscan);
        return -1;
    }
    return 0;
}
```
这个函数很简单，就是用`fscanf`从文件里面读了一个 float 数出来。

#### 1.2 forward

由于支持 slope，而作者将 relu 和带 slope 的 leaky relu 用 if 分开写了，relu 在小于直接给 0，leaky relu 在小于 0 是`*slope`，这里贴出 leaky relu 部分的代码:
```cpp
int ReLU::forward(const Mat& bottom_blob, Mat& top_blob) const {
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    int size = w * h;

    top_blob.create(w, h, channels);
    if (top_blob.empty())
        return -100;

    for (int q=0; q<channels; q++) {
        const float* ptr = bottom_blob.channel(q);
        float* outptr = top_blob.channel(q);

        for (int i=0; i<size; i++) {
            if (ptr[i] < 0)
                outptr[i] = ptr[i] * slope;
            else
                outptr[i] = ptr[i];
        }
    }

    return 0;
}
```
代码很简单，就是开辟一个跟输入大小一样的`ncnn::Mat`，然后逐个通道，逐个位置的元素，按照公式的定义算。

### 2 Pooling 算子

pooling 和 relu 一样，也是一个无权重的算子，只有几个配置参数。pooling 源码在`pooling.h`和`pooling.cpp`这两个文件里面。

#### 2.1 load_param

在 param 文件里面，pooling 层：
```bash
Pooling          pool1            1 1 conv1_relu_conv1 pool1 0 3 2 0 0
```
后面那 5 个 layer 特定参数`0 3 2 0 0`分别代表：pooling_type、kernel_size、stride、pad、global_pooling。

具体的加载源码：
```cpp
int Pooling::load_param(FILE* paramfp) {
    int nscan = fscanf(paramfp, "%d %d %d %d %d", &pooling_type, &kernel_size, &stride, &pad, &global_pooling);
    if (nscan != 5) {
        fprintf(stderr, "Pooling load_param failed %d\n", nscan);
        return -1;
    }
    return 0;
}
```

从文件里面 fscanf 得到 pooling 层特有的五个参数。


#### 2.2 forward

在`forwar`的代码里面，通过`pooling_type`和`global_pooling`这两个变量的控制，一共组合了四种实现：
* 非全局池化：最大值池化(max_pooling)、平均值池化(mean_pooling)
* 全局池化：最大值池化(global_max_pooling)、平均值池化(global_mean_pooling)

实现都是大同小异，这里详细看看全局最大值池化和非全局平均值池化的源码。

##### 2.2.1 全局最大值池化(global_max_pooling)
```CPP
int Pooling::forward(const Mat& bottom_blob, Mat& top_blob) const {
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;

    top_blob.create(1, 1, channels);
    if (top_blob.empty())
        return -100;

    int size = w * h;
    for (int q=0; q<channels; q++){
        const float* ptr = bottom_blob.channel(q);
        float* outptr = top_blob.channel(q);

        float max = ptr[0];
        for (int i=0; i<size; i++) {
            max = std::max(max, ptr[i]);
        }
        outptr[0] = max;
    }
    return 0;
}
```
全局最大值池化就是将原本的`(c,h,w)`的数据，计算每个 chennel 的最大值，把数据做成`(c,1,1)`。

##### 2.2.2 非全局平均值池化
```cpp
int Pooling::forward(const Mat& bottom_blob, Mat& top_blob) const {
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    // 按照用户需要的pad进行padding
    Mat bottom_blob_bordered = bottom_blob;
    if (pad > 0) {
        copy_make_border(bottom_blob, bottom_blob_bordered, pad, pad, pad, pad, BORDER_CONSTANT, 0.f);
        if (bottom_blob_bordered.empty())
            return -100;

        w = bottom_blob_bordered.w;
        h = bottom_blob_bordered.h;
    }

    int outw = (w - kernel_size) / stride + 1;
    int outh = (h - kernel_size) / stride + 1;
    // size跟stride不匹配的话也要padding一下
    int wtail = (w - kernel_size) % stride;
    int htail = (h - kernel_size) % stride;
    if (wtail != 0 || htail != 0) {
        int wtailpad = 0;
        int htailpad = 0;
        if (wtail != 0)
            wtailpad = kernel_size - wtail;
        if (htail != 0)
            htailpad = kernel_size - htail;

        Mat bottom_blob_bordered2;
        copy_make_border(bottom_blob_bordered, bottom_blob_bordered2, 0, htailpad, 0, wtailpad, BORDER_REPLICATE, 0.f);
        if (bottom_blob_bordered2.empty())
            return -100;

        bottom_blob_bordered = bottom_blob_bordered2;
        w = bottom_blob_bordered.w;
        h = bottom_blob_bordered.h;

        if (wtail != 0)
            outw += 1;
        if (htail != 0)
            outh += 1;
    }
    // 创建存结果的ncnn::Mat
    top_blob.create(outw, outh, channels);
    if (top_blob.empty())
        return -100;
    // 计算滑窗内各元素的下标偏移量，主要用于加速索引
    const int maxk = kernel_size * kernel_size;
    std::vector<int> _space_ofs(maxk);
    int* space_ofs = &_space_ofs[0];
    {
        int p1 = 0;
        int p2 = 0;
        int gap = w - kernel_size;
        for (int i = 0; i < kernel_size; i++) {
            for (int j = 0; j < kernel_size; j++) {
                space_ofs[p1] = p2;
                p1++;
                p2++;
            }
            p2 += gap;
        }
    }
    // 正式的pooling计算
    for (int q=0; q<channels; q++) {
        const Mat m(w, h, bottom_blob_bordered.channel(q));
        float* outptr = top_blob.channel(q);

        for (int i = 0; i < outh; i++) {
            for (int j = 0; j < outw; j++) {
                const float* sptr = m.data + m.w * i*stride + j*stride;
                float sum = 0;
                for (int k = 0; k < maxk; k++) {
                    float val = sptr[ space_ofs[k] ];
                    sum += val;
                }
                outptr[j] = sum / maxk;
            }
            outptr += outw;
        }
    }
    return 0;
}
```

过程分四步：
* padding 用户的指定值
* padding 到能计算的 size
* 计算滑窗下标偏移量
* 正式计算

### 3 convolution 算子

卷积的源码位于`convolution.h`和`convolution.cpp`这两个文件。卷积与前面的 relu 和 pooling 不同，他有权重。

#### 3.1 load_param

param 文件里面卷积：
```bash
Convolution      conv1            1 1 data conv1 64 3 1 2 0 1 1728
```

后面的 7 个 layer 特定参数分别是：num_output、kernel_size、dilation、stride、pad、bias_term、weight_data_size。
```cpp
int Convolution::load_param(FILE* paramfp) {
    int nscan = fscanf(paramfp, "%d %d %d %d %d %d %d", &num_output, &kernel_size, &dilation, &stride, &pad, &bias_term, &weight_data_size);
    if (nscan != 7) {
        fprintf(stderr, "Convolution load_param failed %d\n", nscan);
        return -1;
    }

    return 0;
}
```
这个就是 fscanf 一下 7 个参数。


#### 3.2 load_model

卷积有权重，ncnn 中的权重数据都存到了`.bin`文件，
```cpp
int Convolution::load_model(FILE* binfp) {
    int nread;
    // 读取权重的一些基本信息
    union {
        struct {
            unsigned char f0;
            unsigned char f1;
            unsigned char f2;
            unsigned char f3;
        };
        unsigned int tag;
    } flag_struct;
    nread = fread(&flag_struct, sizeof(flag_struct), 1, binfp);
    if (nread != 1) {
        fprintf(stderr, "Convolution read flag_struct failed %d\n", nread);
        return -1;
    }
    unsigned int flag = flag_struct.f0 + flag_struct.f1 + flag_struct.f2 + flag_struct.f3;
    weight_data.create(weight_data_size);
    if (weight_data.empty())
        return -100;

    if (flag_struct.tag == 0x01306B47) { // float16的权重读取，跳过
        // ......
    }
    else if (flag != 0) { // 量化数据的权重读取，跳过
        // ......
    }
    else if (flag_struct.f0 == 0) { // 最原始的float32的读取
        // raw weight data
        nread = fread(weight_data, weight_data_size * sizeof(float), 1, binfp);
        if (nread != 1) {
            fprintf(stderr, "Convolution read weight_data failed %d\n", nread);
            return -1;
        }
    }
    // 有bias项的话也要读bias
    if (bias_term) {
        bias_data.create(num_output);
        if (bias_data.empty())
            return -100;
        nread = fread(bias_data, num_output * sizeof(float), 1, binfp);
        if (nread != 1) {
            fprintf(stderr, "Convolution read bias_data failed %d\n", nread);
            return -1;
        }
    }
    return 0;
}
```
在权重的读取过程中，读取了一个结构体`flag_struct`，这个结构体包含了一些权重的信息，例如是不是 float16 数据，是不是量化了的数据等。只看`float32`的话，代码很简单，就是按照从`.bin`文件里面读到的权重 size 和 bias size，直接从文件里面读指定长度数据就可以了。

#### 3.3 forward
```cpp
int Convolution::forward(const Mat& bottom_blob, Mat& top_blob) const {
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;

    // padding
    Mat bottom_blob_bordered = bottom_blob;
    if (pad > 0) {
        copy_make_border(bottom_blob, bottom_blob_bordered, pad, pad, pad, pad, BORDER_CONSTANT, 0.f);
        if (bottom_blob_bordered.empty())
            return -100;

        w = bottom_blob_bordered.w;
        h = bottom_blob_bordered.h;
    }

    const int kernel_extent = dilation * (kernel_size - 1) + 1;
    int outw = (w - kernel_extent) / stride + 1;
    int outh = (h - kernel_extent) / stride + 1;

    top_blob.create(outw, outh, num_output);
    if (top_blob.empty())
        return -100;

    // 计算kernel各元素的下标偏移量
    const int maxk = kernel_size * kernel_size;
    std::vector<int> _space_ofs(maxk);
    int* space_ofs = &_space_ofs[0];
    {
        int p1 = 0;
        int p2 = 0;
        int gap = w * dilation - kernel_extent;
        for (int i = 0; i < kernel_size; i++) {
            for (int j = 0; j < kernel_size; j++) {
                space_ofs[p1] = p2;
                p1++;
                p2 += dilation;
            }
            p2 += gap;
        }
    }

    // 卷积计算
    const float* weight_data_ptr = weight_data;
    for (int p=0; p<num_output; p++) {
        float* outptr = top_blob.channel(p);
        for (int i = 0; i < outh; i++) {
            for (int j = 0; j < outw; j++) {
                float sum = 0.f;
                if (bias_term)
                    sum = bias_data.data[p]; // 加bias
                const float* kptr = weight_data_ptr + maxk * channels * p;
                // channels
                for (int q=0; q<channels; q++) {
                    const Mat m = bottom_blob_bordered.channel(q);
                    const float* sptr = m.data + m.w * i*stride + j*stride;
                    for (int k = 0; k < maxk; k++) {
                        float val = sptr[ space_ofs[k] ];
                        float w = kptr[k];
                        sum += val * w; // kernel和feature的乘加操作
                    }
                    kptr += maxk;
                }
                outptr[j] = sum;
            }
            outptr += outw;
        }
    }
    return 0;
}
```
这个版本的卷积计算比较原始，没有过多的优化，后续版本的优化比较多，但是没有这个版本的这么方便阅读。

### 4 split 与 concat算子

前面的 relu、pooling、convolution 算子，都是**一路输入一路输出**的算子，split 与 concat 的存在能让网络实现分支的功能。split 是**单路输入多路输出**，concat 是**多路输入单路输出**，这两个算子分别在`split.h`，`split.cpp`和`concat.h`，`concat.cpp`文件里面。在 ncnn 的实现中，split 和 concat 都是无参数无权重的，也就是说只需要从 param 读取 layer 特定参数，不需要从 bin 读取权重参数。


#### 4.1 split

在 param 文件里面，split 这个 layer：
```bash
Split            splitncnn_0      1 2 relu_squeeze1x1 relu_squeeze1x1_splitncnn_0 relu_squeeze1x1_splitncnn_1
```

split 的参数都已经包含在输入输出 blob 信息里面了，一个输入两个输出。

**forward的实现：**
```cpp
int Split::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs) const {
    const Mat& bottom_blob = bottom_blobs[0];
    for (size_t i=0; i<top_blobs.size(); i++) {
        top_blobs[i] = bottom_blob;
    }
    return 0;
}
```
代码非常简洁，就是给输出 blob 每一个都拷贝上输入 blob 的值。从函数的入参可以看到，**单输入单输出的时候**入参是`ncnn::Mat`，而非单输入单输出，入参就是 vector 了。

#### 4.2 concat

在 param 文件里面 concat 的写法：
```bash
Concat           fire9/concat     2 1 relu_expand1x1 relu_expand3x3 fire9/concat
```
**forward的实现：**
```cpp
int Concat::forward(const std::vector<Mat>& bottom_blobs, std::vector<Mat>& top_blobs) const {
    int w = bottom_blobs[0].w;
    int h = bottom_blobs[0].h;

    // total channels
    int top_channels = 0;
    for (size_t b=0; b<bottom_blobs.size(); b++) {
        const Mat& bottom_blob = bottom_blobs[b];
        top_channels += bottom_blob.c;
    }
    Mat& top_blob = top_blobs[0];
    top_blob.create(w, h, top_channels);
    if (top_blob.empty())
        return -100;
    int q = 0;
    for (size_t b=0; b<bottom_blobs.size(); b++) {
        const Mat& bottom_blob = bottom_blobs[b];
        int channels = bottom_blob.c;
        int size = bottom_blob.cstep * channels;
        const float* ptr = bottom_blob;
        float* outptr = top_blob.channel(q);
        for (int i=0; i<size; i++) {
            outptr[i] = ptr[i];
        }
        q += channels;
    }
    return 0;
}
```

*ncnn 默认从 channel 维度做 concat。*
