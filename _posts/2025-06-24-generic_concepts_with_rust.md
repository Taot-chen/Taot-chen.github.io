---
layout: post
title: generic_concepts_with_rust
date: 2025-06-24
tags: [rust]
author: taot
---


## Rust 通用编程概念

变量、基本类型、函数、注释和控制流。


### 1 变量和可变性

Rust 中默认情况下变量是**不可变的**（immutable），也可以选择让变量是可变的（mutable）。当变量不可变时，一旦一个值绑定到一个变量名后，就不能更改该值了。

实验变量的特性：
```bash
cargo new variables
cd variables
```

将 `src/main.rs` 替换成以下代码:
```rust
fn main() {
    let x = 5;
    println!("The value of x is: {}", x);
    x = 6;
    println!("The value of x is: {}", x);
}
```

这段代码编译是无法通过的，因为代码中对不可变变量 x 的值做了修改。

```bash
cargo run

# 输出
error[E0384]: cannot assign twice to immutable variable `x`
 --> src/main.rs:4:5
  |
2 |     let x = 5;
  |         - first assignment to `x`
3 |     println!("The value of x is: {}", x);
4 |     x = 6;
  |     ^^^^^ cannot assign twice to immutable variable
  |
help: consider making this binding mutable
  |
2 |     let mut x = 5;
  |         +++

For more information about this error, try `rustc --explain E0384`.
error: could not compile `variables` (bin "variables") due to 1 previous error
```

上面的错误指出错误的原因是`cannot assign twice to immutable variable x`（不能对不可变变量二次赋值）。当我们尝试改变一个前面指定为不可变的值时我们会得到编译期错误，


如果我们代码的一部分假设某个值永远不会更改，而代码的另一部分更改了该值，那很可能第一部分代码以不可意料的方式运行。这个 bug 的根源在实际开发中可能很难追踪，特别是第二部分代码只是偶尔变更了原来的值。**Rust 编译器保证了当我们声明了一个值不会改变时，那它就真的不可改变**。

可以通过在变量名前加上 `mut` 使得它们可变。增加 `mut` 的操作还向以后的读代码的人传达了代码的其他部分将会改变这个变量值。

将 src/main.rs 改为以下内容：

```rust
fn main() {
    let mut x = 5;
    println!("The value of x is: {}", x);
    x = 6;
    println!("The value of x is: {}", x);
}
```

此时代码就可以正常编译运行了。

> 这里的设计考虑可能是这样的：在使用大型数据结构的情形下，在同一位置更改实例可能比复制并返回新分配的实例要更快。使用较小的数据结构时，通常创建新的实例并以更具函数式编程的风格来编写程序，可能会更容易理解，所以值得以较低的性能开销来确保代码清晰。


#### 1.1 常量

常量（constant）是绑定到一个常量名且不允许更改的值。常量具有如下特点：

* 常量不允许使用 mut。常量不仅仅默认不可变，而且自始至终不可变；
* 常量使用 `const` 关键字而不是 `let` 关键字来声明，并且值的类型必须注明;
* 常量可以在任意作用域内声明，包括全局作用域;
* 常量只能设置为常量表达式，而不能是函数调用的结果或是只能在运行时计算得到的值。

声明一个常量：

```rust
const THREE_HOURS_IN_SECONDS: u32 = 60 * 60 * 3;
```

常量名为 `THREE_HOURS_IN_SECONDS`，值设置为 60（一分钟内的秒数）乘以 60（一小时内分钟数）再乘以 3。

Rust 常量的命名约定是全部字母都使用大写，并使用下划线分隔单词。

在声明的作用域内，常量在程序运行的整个过程中都有效。对于应用程序域中程序的多个部分可能都需要知道的值的时候，常量是一个很有用的选择。将整个程序中用到的硬编码（hardcode）值命名为常量，对于将该值的含义传达给代码的未来维护者很有用。


#### 1.2 遮蔽

可以声明和前面变量具有相同名称的新变量, 这个是第一个变量被第二个变量遮蔽（shadow），这意味着当我们使用变量时我们看到的会是第二个变量的值。可以通过使用相同的变量名并重复使用 `let` 关键字来遮蔽变量:

`shadow.rs`

```rust
fn main() {
    let x = 5;
    let x = x + 1;
    {
        let x = x * 2;
        println!("The value of x in the inner scope is: {}", x);
    }
    println!("The value of x is: {}", x);
}
```

首先将数值 5 绑定到 x。然后通过重复使用`let x =`来遮蔽之前的 x，并取原来的值加上 1，所以 x 的值变成了 6。在内部作用域内，第三个 `let` 语句同样遮蔽前面的 x，取之前的值并乘上 2，得到的 x 值为 12。当该作用域结束时，内部遮蔽结束并且 x 恢复成 6:

```bash
rustc shadow.rs
./shadow

The value of x in the inner scope is: 12
The value of x is: 6
```

遮蔽和将变量标记为 `mut` 的方式不同:
* 除非我们再次使用 `let` 关键字，否则若是我们不小心尝试重新赋值给这个变量，我们将得到一个编译错误。
* 在再次使用 `let` 关键字时有效地创建了一个新的变量，所以我们可以改变值的类型，但重复使用相同的名称。

例如，使用变量遮蔽，下面的代码是合法的：
```rust
let spaces = "   ";
let spaces = spaces.len();
```

第一个 spaces 变量是一个字符串类型，第二个 spaces 变量是一个数字类型。

但是如果使用 `mut` 如下代码是不合法的：
```rust
let mut spaces = "   ";
spaces = spaces.len();
```




### 2 数据类型

Rust 是一种静态类型（statically typed）的语言，它必须在编译期知道所有变量的类型。


#### 2.1 标量类型

标量（scalar）类型表示单个值。Rust 有 4 个基本的标量类型：整型、浮点型、布尔型和字符。


##### 2.1.1 整数类型

整数（integer）是没有小数部分的数字。

|长度|有符号类型|无符号类型|
|---|---|---|
|8 位|i8|u8|
|16 位|i16|u16|
|32 位|i32|u32|
|64 位|i64|u64|
|128 位|i128|u128|
|arch|isize|usize|

有符号和无符号表示数字能否取负数，有符号数字以二进制补码形式存储。

数字表示范围与 C 语言定义的相同。每个有符号类型规定的数字范围是 $-(2^n - 1) \rightarrow 2^{n - 1} - 1$ ，无符号类型可以存储的数字范围是 $0 \rightarrow 2^n - 1$

`isize` 和 `usize` 类型取决于程序运行的计算机体系结构，若使用 64 位架构系统则为 64 位，若使用 32 位架构系统则为 32 位。

Rust 整型默认是 `i32`。`isize` 和 `usize` 的主要应用场景是用作某些集合的索引。

**Rust 针对整型溢出的处理**：
* 使用 `wrapping_*` 方法在所有模式下进行包裹，例如 wrapping_add
* 如果使用 `checked_*` 方法时发生溢出，则返回 None 值
* 使用 `overflowing_*` 方法返回该值和一个指示是否存在溢出的布尔值
* 使用 `saturating_*` 方法使值达到最小值或最大值


##### 2.1.2 浮点类型

浮点数（floating-point number）是带有小数点的数字，Rust 的浮点型是 `f32` 和 `f64`，它们的大小分别为 32 位（单精度）和 64 位（双精度）。默认浮点类型是 `f64`。



**数字运算**

Rust 的所有数字类型都支持基本数学运算：加法、减法、乘法、除法和取模运算。整数除法会向下取整。



##### 2.1.3 布尔类型

Rust 中的布尔类型也有两个可能的值：`true` 和 `false`。布尔值的大小为 1 个字节。Rust 中的布尔类型使用 `bool` 声明。

```rust
let t = true;
let f: bool = false; // with explicit type annotation
```

使用布尔值的主要地方是条件判断，如 `if` 表达式。


##### 2.1.4 字符类型

Rust 的 `char`（字符）类型是该语言最基本的字母类型:

```rust
let c = 'z';
let z = 'ℤ';
let heart_eyed_cat = '😻';
```

声明的 `char` 字面量采用**单引号**括起来，字符串字面量是用**双引号**括起来。

Rust 的字符类型大小为 4 个字节，表示的是一个 Unicode 标量值。标音字母，中文/日文/韩文的文字，emoji，还有零宽空格(zero width space)在 Rust 中都是合法的字符类型。



#### 2.2 复合类型

复合类型（compound type）可以将多个值组合成一个类型。Rust 有两种基本的复合类型：元组（tuple）和数组（array）。



##### 2.2.1 元组类型

元组是将多种类型的多个值组合到一个复合类型中的一种基本方式。元组的长度是固定的：声明后，它们就无法增长或缩小。

通过在小括号内写入以逗号分隔的值列表来创建一个元组。

元组中的每个位置都有一个类型，并且元组中不同值的类型不要求是相同的。

```rust
let tup: (i32, f64, u8) = (500, 6.4, 1);
```

变量 `tup` 绑定到整个元组，元组被认作是单个复合元素。 想从元组中获取个别值，可以使用模式匹配来解构（destructure）元组的一个值，

```rust
let tup = (500, 6.4, 1);
let (x, y, z) = tup;
println!("The value of y is: {}", y);
```

首先创建一个元组并将其绑定到变量 `tup` 上。 然后它借助 `let` 来使用一个模式匹配 `tup`，并将它分解成三个单独的变量 `x`、`y` 和 `z`。 这过程称为解构（destructuring）。

除了通过模式匹配进行解构外，还可以使用一个句点（`.`）连上要访问的值的索引来直接访问元组元素:

```rust
let x: (i32, f64, u8) = (500, 6.4, 1);
let five_hundred = x.0;
let six_point_four = x.1;
let one = x.2;
```

创建一个元组 `x`，然后通过使用它们的索引为每个元素创建新的变量。元组中的第一个索引为 0。

没有任何值的元组 `()` 是一种特殊的类型，只有一个值，也写成 `()`。该类型被称为单元类型（unit type），该值被称为单元值（unit value）。如果表达式不返回任何其他值，就隐式地返回单元值。



##### 2.2.2 数组类型

数组的每个元素必须具有相同的类型。Rust 中的数组具有固定长度。

在方括号内以逗号分隔的列表形式将值写到数组中：

```rust
let a = [1, 2, 3, 4, 5];
```

当希望将数据分配到栈（stack）而不是堆（heap）时，或者希望确保始终具有固定数量的元素时，数组特别有用。它们不像 vector 类型那么灵活。


使用方括号编写数组的类型，其中包含每个元素的类型、分号，然后是数组中的元素数

```rust
let a: [i32; 5] = [1, 2, 3, 4, 5];
```

这里，`i32` 是每个元素的类型。分号之后，数字 5 表明该数组包含 5 个元素。


如果要为每个元素创建包含相同值的数组，可以指定初始值，后跟分号，然后在方括号中指定数组的长度，

```rust
let a = [3; 5];
```

变量名为 `a` 的数组将包含 5 个元素，这些元素的值初始化为 3。这种写法与 `let a = [3, 3, 3, 3, 3];` 效果相同，但更简洁。


**访问数组元素**

数组是可以在栈上分配的已知固定大小的单个内存块。可以使用索引访问数组的元素

```rust
let a = [1, 2, 3, 4, 5];
let first = a[0];
let second = a[1];
```


**无效的数组元素访问**

如果尝试月结访问数组会怎样？

```rust
use std::io;

fn main() {
    let a = [1, 2, 3, 4, 5];
    println!("Please enter an array index.");
    let mut index = String::new();

    io::stdin()
        .read_line(&mut index)
        .expect("Failed to read line");

    let index: usize = index
        .trim()
        .parse()
        .expect("Index entered was not a number");

    let element = a[index];
    println!(
        "The value of the element at index {} is: {}",
        index, element
    );
}
```
代码编译成功。如果使用 `cargo run` 来运行此代码并输入 0、1、2、3 或 4，则程序将打印数组对应索引的值。如果输入的是超出数组末尾的数字，例如 10，则会看到类似以下的输出：

```bash
thread 'main' panicked at src/main.rs:17:19:
index out of bounds: the len is 5 but the index is 10
note: run with `RUST_BACKTRACE=1` environment variable to display a backtrace
```

该程序在索引操作中使用无效值时导致运行时（runtime）错误。当你尝试使用索引访问元素时，Rust 将检查你指定的索引是否小于数组长度。如果索引大于或等于数组长度，Rust 会出现 panic。这种检查在运行时进行，因为编译器可能无法知道用户之后运行代码时将输入什么值。

很多其他语言（例如C语言），在你使用不正确的索引时，可以访问无效的内存并继续运行程序，Rust 通过立即退出来的方式防止这种错误。

