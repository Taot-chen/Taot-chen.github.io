---
layout: post
title: rust_get_started
date: 2025-06-23
tags: [rust]
author: taot
---


## Rust Get Started

### 0 Rust

Rust 最早是 Mozilla 雇员 Graydon Hoare 的个人项目。从 2009 年开始，得到了 Mozilla 研究院的资助，2010 年项目对外公布，2010 ～ 2011 年间实现自举。自此以后，Rust 在部分重构 -> 崩溃的边缘反复横跳（历程极其艰辛），终于，在 2015 年 5 月 15 日发布 1.0 版。

**1）学习曲线陡峭**

* 实践中如何融会贯通的运用
* 遇到了坑时（生命周期、借用错误，自引用等）如何迅速、正确的解决
* 大量的标准库方法记忆及熟练使用，这些是保证开发效率的关键
* 心智负担较重，特别是初中级阶段

**2）运行效率高**

各种零开销抽象、深入到底层的优化潜力、优质的标准库和第三方库实现，Rust 具备非常优秀的性能，和 C、C++ 是 一个级别。

只要按照正确的方式使用 Rust，无需性能优化，就能有非常优秀的表现。（这是不是 rust 的编译器写的比较好？）


**3）开发效率**

在不熟悉标准库、生命周期和所有权的常用解决方法的阶段，开发速度很慢；熟悉之后，能够迅速写出高质量、安全、高效的代码。



Rust 笨版本迭代比较频繁，每 6 周发布一个迭代版本，2-3 年发布一个新的大版本。


### 1 安装 Rust


#### 1.1 通过 rustup 来下载 Rust

rustup 是 Rust 的安装程序，也是它的版本管理程序。使用 rustup 来安装 Rust。

```bash
curl --proto '=https' --tlsv1.2 https://sh.rustup.rs -sSf | sh
```

这个命令将下载一个脚本并开始安装 rustup 工具，此工具将安装 Rust 的最新稳定版本。后面的操作正常操作即可，直到出现下面这行，说明安装成功：

```bash
Rust is installed now. Great!
```

ubuntu 系统默认自带了 C 编译器，但是确保有有效地 linker 可用，可以执行下面的命令来安装`build-essential`:

```bash
sudo apt install build-essential
```

#### 1.2 检查安装是否成功

```bash
rustc -V
rustc 1.84.0 (9fc6b4312 2025-01-07)

cargo -V
# cargo 1.84.0 (66221abde 2024-11-19)
```

*注：安装完之后会需要重新连接终端，或者更新 shell 的环境变量*


#### 1.3 更新与卸载

```bash
# 更新
rustup update

# 卸载
rustup self uninstall
```

安装 Rust 的同时也会在本地安装一个文档服务，运行`rustup doc`可以在浏览器打开本地文档。



### 1.4 VsCode Rust 环境配置


VSCode 的 Rust 插件

* rust-analyzer，这个必装
* rust，这个不太好用，可以不要

其他插件：

* Even Better TOML，支持 .toml 文件完整特性
* Error Lens, 更好的获得错误展示
* CodeLLDB, Debugger 程序



### 2 Hello World

#### 2.1 打印 hello world

Rust 不关心代码存放的位置，可以把代码集中管理。

我这里集中放置在 practice 下面：
```bash
mkdir -p practice/hello_world && cd practice/hello_world
```

创建一个源文件并命名为 `main.rs`。Rust 文件通常以 `.rs` 扩展名结尾。

`main.rs`

```rust
fn main() {
    println!("Hello, world!");
}
```

编译并运行文件：

```bash
rustc main.rs
./main
# Hello, world!
```


#### 2.2 Hello world 理解

**1） 主函数定义**

```rust
fn main() {
}
```

定义了 rust 的 main 函数。main 函数比较特殊，**它始终是每个可执行 Rust 程序中运行的第一个代码**。第一行声明一个名为 main 的函数，不带参数也没有返回值。如果有参数，参数名放置在`()`内。函数主体用大括号 {} 括起来。



**2）主函数函数体**

```rust
    println!("Hello, world!");
```

说明：

* Rust 风格的缩进使用 4 个空格，而不是制表符。
* `println!` 调用 Rust 宏。如果改为调用函数，则应该将其输入为 `println`（不含 `!`）。当看到一个 `!`，则意味着调用的是宏而不是普通的函数。
* 用分号（`;`，注意这是**英文分号**）结束该行，这表明该表达式已结束，下一个表达式已准备好开始。Rust 代码的大多数行都以一个` ; `结尾


在前面的过程中也不难看出，Rust 是一种**编译型编程语言，编译和运行是独立的步骤**。



### 3 Cargo

使用`rustc`编译对简单的程序可以轻松胜任，但随着项目的增长，想要管理项目中所有相关内容，并想让其他用户和项目能够容易共享你的代码，就需要用到 Rust 的构建系统和包管理器 Cargo。

Cargo 是 Rust 的构建系统和包管理器，他可以处理很多任务，比如构建代码、下载依赖库，以及编译这些库。

前面安装 rust 的时候，就已经自带了 cargo，可以查看 cargo 的版本：

```bash
cargo --version

cargo 1.84.0 (66221abde 2024-11-19)
```



#### 3.1 使用 cargo 创建项目

```bash
cd practice
cargo new hello_cargo && cd hello_cargo
```

`cargo new hello_cargo`命令新建了一个名为`hello_cargo`的目录, 将项目命名为`hello_cargo`，同时 Cargo 在一个同名目录中创建项目文件。

`hello_cargo`的目录结构如下：

```bash
.
├── Cargo.toml
└── src
    └── main.rs
```

如果运行命令的目录不在现有的 Git 仓库中，那么 cargo 还会在 `hello_cargo` 目录初始化了一个 Git 仓库，并带有一个 `.gitignore` 文件。如果在现有的 Git 仓库中运行 cargo new，则不会生成 Git 文件，我这里就没有生成 git 文件。


**1）Cargo.toml**

```bash
[package]
name = "hello_cargo"
version = "0.1.0"
edition = "2021"

[dependencies]
```

此文件使用 TOML (Tom's Obvious, Minimal Language) 格式，这是 Cargo 配置文件的格式

* `[package]`，是一个表块（section）标题，表明下面的语句用来配置一个包（package）
    接下来的三行设置了 Cargo 编译程序所需的配置：项目的名称、版本，以及使用的 Rust 大版本号（edition，区别于 version）。Rust 的核心版本，即 2015、2018、2021 版等。
* `[dependencies]`，罗列项目依赖的表块，在 Rust 中，代码包被称为 crate。这个项目并不需要其他的 crate


**2）src/main.rs **

```rust
fn main() {
    println!("Hello, world!");
}
```

Cargo 生成的一个`Hello, world!`程序，和前面的 hello_world 的代码一致。

到目前为止，之前项目与 Cargo 生成项目的区别是，Cargo 将代码放在 src 目录，同时项目根目录包含一个 Cargo.toml 配置文件。

Cargo 期望源文件存放在`src`目录中。项目根目录只存放说明文件（README）、许可协议（license）信息、配置文件和其他跟代码无关的文件。


#### 3.2 构建并运行 cargo 项目

**1）cargo build**

```bash
cd hello_cargo
cargo build

Compiling hello_cargo v0.1.0
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 0.27s
```

这个命令会创建一个可执行文件`target/debug/hello_cargo`：

```bash
./target/debug/hello_cargo
```

终端上应该会打印出`Hello, world!`。首次运行`cargo build`时，也会使 Cargo 在项目根目录创建一个新文件`Cargo.lock`。这个文件记录项目依赖的实际版本。这个文件不需要自己去修改，交给 cargo 管理即可。

也可以使用`cargo run`在一个命令中同时编译代码并运行生成的可执行文件：


**2）cargo run**

```bash
cargo run

Finished `dev` profile [unoptimized + debuginfo] target(s) in 0.00s
     Running `target/debug/hello_cargo`
Hello, world!
```

这里的 log 里面没有出现 Cargo 正在编译`hello_cargo`的输出，是因为 Cargo 发现文件并没有被改变，就直接运行了二进制文件。


**3）cargo check**

Cargo 还提供了一个名为 `cargo check` 的命令。该命令快速检查代码确保其可以编译，但并不产生可执行文件：

```bash
Checking hello_cargo v0.1.0
Finished `dev` profile [unoptimized + debuginfo] target(s) in 0.05s
```

通常`cargo check`要比`cargo build`快得多，它省略了生成可执行文件的步骤。如果在编写代码时持续的进行检查`cargo check`会加速开发！编写代码时定期运行`cargo check`确保它们可以编译。当准备好使用可执行文件时才运行`cargo build`。


#### 3.3 发布构建

当项目最终准备好发布时，可以使用`cargo build --release`来优化编译项目。会在`target/release`下生成可执行文件。这些优化可以让 Rust 代码运行的更快，不过启用这些优化也需要消耗更长的编译时间。

```bash
 Compiling hello_cargo v0.1.0
    Finished `release` profile [optimized] target(s) in 0.17s
```


对于简单项目， Cargo 并不比 rustc 提供了更多的优势，但是随着项目复杂度的提高，对于拥有多个 crate 的复杂项目 cargo 的优势就凸显出来了。
