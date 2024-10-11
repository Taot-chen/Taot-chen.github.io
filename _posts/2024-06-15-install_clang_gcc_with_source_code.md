---
layout: post
title: install_clang_gcc_with_source_code
date: 2024-06-15
tags: [tools]
author: taot
---

## 源码编译安装 clang/gcc

在同一个机器上有多个人同时使用的时候，机器的环境比较复杂，会存在和别人使用的基础工具版本不同的情况，这个时候为了不影响其他人，也不被其他人影响，可以通过使用源码来编译。编译完成之后，通过环境变量的设置，来使其只针对当前的用户或者 teminal 生效。

### 1 源码编译安装 clang

这里以从源码安装`clang-10`为例。

* 获取 clang 的源码
      Clang 的源码位于 llvm 源码目录`llvm-project/clang`中，因此直接获取 llvm 的源码
      ```bash
      git clone https://github.com/llvm/llvm-project.git
      ```
    * 切换到指定的版本（optional）
      ```bash
      git checkout release/10.x
      ```
    * 安装构建工具
      构建工具可以是`GNU make`、`ninja`等。为了更快的增量编译，Clang 官网推荐使用`ninja`。
      ```bash
      sudo apt-get install ninja-build
      ```
    * 调整配置参数（optional）
      * 可以调整配置参数，以获取定制的编译器。
        默认情况下，所有可支持的目标机器的相关代码都会被编译。为了减少编译时间，我们可以只编译可能会用到的目标机器。如果要指定只编译的目标机器为X86、AArch64和RISCV。可以使用如下选项：
        ```bash
        -DLLVM_TARGETS_TO_BUILD="X86;AArch64;RISCV"
        ```
      * 定编译哪些项目（optional）
        如果要指定只编译的项目为clang。可以使用如下选项：
        ```bash
        -DLLVM_ENABLE_PROJECTS=clang
        ```
        目前，所有的可选项如下所示：
        ```bash
        clang;clang-tools-extra;compiler-rt;debuginfo-tests;libc;libclc;libcxx;libcxxabi;libunwind;lld;lldb;mlir;openmp;parallel-libs;polly;pstl;flang
        ```

      * 指定生成的版本是 DEBUG 还是 RELEASE（optional）
        默认情况下，生成的是DEBUG版本。如果要指定生成RELEASE版本，可以使用如下选项：
        ```bash
        -DCMAKE_BUILD_TYPE=Release
        ```
      * 指定生成静态库还是共享库（optional）
        默认情况下，生成的是静态库。如果要指定生成共享库，可以使用如下选项：
        ```bash
        -DBUILD_SHARED_LIBS=ON
        ```
    * 编译
      * 创建并切到 build 目录
        在 llvm-project 目录中执行如下命令：
        ```bash
        mkdir build && cd build
        ```
      * 仅编译 Clang
        在 llvm-project/build 目录中执行如下命令：
        ```bash
        cmake -DLLVM_ENABLE_PROJECTS=clang -DBUILD_SHARED_LIBS=ON -G Ninja ../llvm/
        ninja clang -j128
        ```
        *选项-j128表示并发任务最多可以有 128 个，从而减少编译时间。实际数量可根据机器实际的逻辑处理器数量来设置。*
    * 测试
      * 安装 Clang 测试套件的依赖
        ```bash
        sudo apt-get install python3-distutils
        ```
        *在不同的机器上运行时，缺少的依赖可能不同。如果运行 Clang 测试套件失败，则根据报错内容进行相应地解决。*
      * 运行 Clang 测试套件
        在 llvm-project/build 目录中执行如下命令：
        ```bash
        ninja check-clang
        ```
        测试完成之后会输出测试结果。
    **编译的 Clang——clang-10.0.0 位于 llvm-project/build/bin 目录中。**


### 2 源码编译安装 gcc

这里以从源码安装`gcc-9.3.0`为例。

* 源码下载
      ```bash
      wget http://ftp.gnu.org/gnu/gcc/gcc-9.3.0/gcc-9.3.0.tar.gz
      ```
    * 编译安装
      ```bash
      tar -zxvf gcc-9.3.0.tar.gz # 解压压缩包
      cd gcc-9.3.0
      mkdir build
      ../configure  # 开始配置安装
      ```
      
      开始配置后可能会出现缺少缺少 GMP、MPFR和 MPC这三个高精度的数学库的问题如
      ```bash
      configure: error: Building GCC requires GMP 4.2+, MPFR 3.1.0+ and MPC 0.8.0+.
      Try the --with-gmp, --with-mpfr and/or --with-mpc options to specify
      their locations. 
      ```
      
      安装缺少的库：
      ```bash
      sudo apt-get install libgmp-dev libmpfr-dev libmpc-dev gcc-multilib g++-multilib 
      ```

      再次使用 `../configure --enable-multilib` 构建安装环境，构建完成后得到一个Makefile文件。开始编译
      ```bash
      make -j128  # -j选项后的数字取决于机器的逻辑处理器数量
      ```

      等待安装完成之后，输入`sudo make install`安装gcc。

    编译完成的`gcc-9.3.0`位于`gcc-9.3.0/build/gcc`目录中。


### 3 配置环境变量

在 `~/.bashrc`末尾添加如下内容：
```bash
export PATH=/path/to/gcc-9.3.0/build/gcc:$PATH
export PATH=/path/to/llvm-project/build/bin:$PATH
```
文件修改保存之后，`source ~/.bashrc` 使配置生效。

如果后面不需要使用这两个对应版本的工具了，重新设置环境变量即可。
