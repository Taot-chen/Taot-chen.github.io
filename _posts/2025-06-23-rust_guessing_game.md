---
layout: post
title: rust_guessing_game
date: 2025-06-23
tags: [rust]
author: taot
---


## guessing game

通过一个经典的新手编程问题，猜数字游戏来熟悉`let`、`match`、方法、关联函数、引用外部 crate 等知识。

程序会随机生成一个 1 到 100 之间的随机整数。接着它会请玩家猜一个数并输入，然后提示猜测是大了还是小了。如果猜对了，它会打印祝贺信息并退出。


### 1 创建新项目

```bash
cargo new guessing_game
cd guessing_game
```

`cargo new` 命令生成了一个 `Hello, world!` 程序。


### 2 处理一次猜测

猜数字程序的第一步，请求用户输入，处理该输入，并检查输入是否符合预期格式。

```rust
use std::io;

fn main() {
    println!("Guess the number!");
    println!("Please input your guess.");

    let mut guess = String::new();

    io::stdin()
        .read_line(&mut guess)
        .expect("Failed to read line");

        println!("You guessed: {}", guess);
}
```

说明：

* 为了获取用户输入并打印结果作为输出，需要引入`io`输入/输出库到当前作用域。`io`库来自于标准库(`std::`)
    * 默认情况下，Rust会将少量标准库中定义的程序项（item）引入到每个程序的作用域中。这些项称作 `prelude`。如果需要的类型不在 prelude 中，必须使用 use 语句显式地将其引入作用域。

* 创建一个储存用户输入的变量（variable）
    ```rust
    let mut guess = String::new();
    ```

    * 使用`let`声明来创建变量
    * 在 Rust 中，变量默认是不可变的
    * 在变量名前使用`mut`来使一个变量可变
        ```rust
        let apples = 5; // 不可变
        let mut bananas = 5; // 可变
        ```

        * // 语法开始一个注释，持续到行尾
    因此，` let mut guess = String::new();` 引入一个叫做`guess`的可变变量。等号（`=`）告诉 Rust 现在想将某个值绑定在变量上。等号的右边是 `guess` 所绑定的值，它是 `String::new` 的结果，这个函数会返回一个 `String` 的新实例。

    * `::new` 那一行的 `::` 语法表明 `new` 是 `String` 类型的一个关联函数
        * **关联函数**（associated function）是实现一种特定类型的函数，在这个例子中类型是 `String`。
        * `new` 函数创建了一个新的空字符串。很多类型上有 `new` 函数，因为它是创建类型实例的惯用函数名。

* 处理用户输入的 `io` 库中的函数 `stdin`
    ```rust
    io::stdin()
        .read_line(&mut guess)
    ```
    * 如果程序的开头没有使用 `use std::io` 引入 `io` 库，我们仍可以通过把函数调用写成 `std::io::stdin` 来使用函数。`stdin` 函数返回一个 `std::io::Stdin` 的实例，这代表终端标准输入句柄的类型。

    * `.read_line(&mut guess)` 这一行调用了 `read_line` 方法从标准输入句柄获取用户输入。将 `&mut guess` 作为参数传递给 `read_line()`, 告诉它在哪个字符串存储用户输入。
        * `read_line`，无论用户在标准输入中键入什么内容，都将其存入一个字符串中（不覆盖其内容），所以它需要字符串作为参数。这个字符串参数需要是可变的，以便该方法可以更改字符串的内容。
        * `&` 表示这个参数是一个引用（reference），它允许多处代码访问同一处数据，而无需在内存中多次拷贝。引用默认是不可变的，因此，需要写成 `&mut guess` 来使其可变

* 使用 Result 类型来处理潜在的错误
    ```rust
    .expect("Failed to read line");
    ```

    * `read_line`会有一个返回值，在这里是返回 `io::Result`，`Result` 类型是 枚举（enumerations），通常也写作 `enum`。枚举类型持有固定集合的值，这些值被称为枚举的成员（variant）。枚举往往与条件表达式 `match` 一起使用，可以方便地根据枚举值是哪个成员来执行不同的代码。
        * `Result` 的成员是 `Ok` 和 `Err`，`Ok` 成员表示操作成功，内部包含成功时产生的值。`Err` 成员则意味着操作失败，并且包含失败的前因后果。
        * `io::Result` 的实例拥有 `expect` 方法。如果 `io::Result` 实例的值是 `Err`，`expect` 会导致程序崩溃，并显示当做参数传递给 `expect` 的信息。如果 `read_line` 方法返回 `Err`，则可能是来源于底层操作系统错误的结果。如果 `io::Result` 实例的值是 `Ok`，`expect` 会获取 `Ok` 中的值并原样返回。这里这个值是用户输入的字节数。

        * 如果不调用 expect，程序也能编译，不过会出现一个警告：
            ```bash
            warning: unused `Result` that must be used
            --> src/main.rs:10:5
            |
            10 |     io::stdin().read_line(&mut guess);
            |     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            |
            = note: `#[warn(unused_must_use)]` on by default
            = note: this `Result` may be an `Err` variant, which should be handled

            warning: `guessing_game` (bin "guessing_game") generated 1 warning
                Finished dev [unoptimized + debuginfo] target(s) in 0.59s
            ```
            Rust 警告我们没有使用 read_line 的返回值 Result，说明有一个可能的错误没有处理。消除警告的正确做法是实际编写错误处理代码。
        
* 使用 println! 占位符打印值

    ```rust
    println!("You guessed: {}", guess);
    ```
    这行代码现在打印了存储用户输入的字符串。里面的 `{}` 是预留在特定位置的占位符。使用 `{}` 也可以打印多个值：第一对 `{}` 使用格式化字符串之后的第一个值，第二对则使用第二个值，依此类推。
    ```rust
        let x = 5;
        let y = 10;
        println!("x = {} and y = {}", x, y);
    ```

* 测试这部分代码

```bash
cargo run

 Compiling guessing_game v0.1.0
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 0.28s
     Running `target/debug/guessing_game`
Guess the number!
Please input your guess.
4
You guessed: 4
```

这部分代码行为符合预期。



### 3 生成一个随机数字

Rust 标准库中尚未包含随机数功能。然而，Rust 团队提供了一个包含上述功能的 rand crate。


#### 3.1 使用 crate 来增加更多功能

crate 是一个 Rust 代码包。我们正在构建的项目是一个 二进制 crate，它生成一个可执行文件。

rand crate 是一个 库 crate，库 crate 可以包含任意能被其他程序使用的代码，但是不能独自执行。

在使用 `rand` 编写代码之前，需要修改 `Cargo.toml` 文件，引入一个 rand 依赖。在`Cargo.toml` 文件最后面的 `[dependencies]` 表块标题之下添加如下内容：

```bash
rand = "0.8.3"
```

在 `Cargo.toml` 文件中，表头以及之后的内容属同一个表块，直到遇到下一个表头才开始新的表块。在 `[dependencies]`表块中，需要告诉 Cargo 本项目依赖了哪些外部 crate 及其版本。

Cargo 理解语义化版本（Semantic Versioning，有时也称为 SemVer），这是一种定义版本号的标准。`0.8.3` 实际上是 `^0.8.3` 的简写，它表示任何至少包含 `0.8.3` 但低于 `0.9.0` 的版本。 Cargo 认为这些版本具有与` 0.8.3` 版本兼容的公有 API， 并且此规范可确保你将获得最新的补丁版本。

重新构建项目

```bash
cargo build

Updating crates.io index
     Locking 15 packages to latest compatible versions
      Adding byteorder v1.5.0
      Adding cfg-if v1.0.0
      Adding getrandom v0.2.15
      Adding libc v0.2.169
      Adding ppv-lite86 v0.2.20
      Adding proc-macro2 v1.0.93
      Adding quote v1.0.38
      Adding rand v0.8.5
      Adding rand_chacha v0.3.1
      Adding rand_core v0.6.4
      Adding syn v2.0.96
      Adding unicode-ident v1.0.14
      Adding wasi v0.11.0+wasi-snapshot-preview1
      Adding zerocopy v0.7.35
      Adding zerocopy-derive v0.7.35
  Downloaded getrandom v0.2.15
  Downloaded quote v1.0.38
  Downloaded ppv-lite86 v0.2.20
  Downloaded cfg-if v1.0.0
  Downloaded rand_chacha v0.3.1
  Downloaded rand_core v0.6.4
  Downloaded proc-macro2 v1.0.93
  Downloaded byteorder v1.5.0
  Downloaded zerocopy-derive v0.7.35
  Downloaded unicode-ident v1.0.14
  Downloaded rand v0.8.5
  Downloaded zerocopy v0.7.35
  Downloaded syn v2.0.96
  Downloaded libc v0.2.169
  Downloaded 14 crates (1.6 MB) in 1.99s
   Compiling proc-macro2 v1.0.93
   Compiling unicode-ident v1.0.14
   Compiling libc v0.2.169
   Compiling cfg-if v1.0.0
   Compiling byteorder v1.5.0
   Compiling quote v1.0.38
   Compiling getrandom v0.2.15
   Compiling syn v2.0.96
   Compiling rand_core v0.6.4
   Compiling zerocopy-derive v0.7.35
   Compiling zerocopy v0.7.35
   Compiling ppv-lite86 v0.2.20
   Compiling rand_chacha v0.3.1
   Compiling rand v0.8.5
   Compiling guessing_game v0.1.0
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 7.74s
```


当我们引入了一个外部依赖后，Cargo 将从 registry 上获取所有依赖所需的最新版本，这是一份来自 [Crates.io](https://crates.io/) 的数据拷贝。Crates.io 是 Rust 生态环境中开发者们向他人贡献 Rust 开源项目的地方。

在更新完 registry 后，Cargo 检查 `[dependencies]` 表块并下载缺失的 crate 。这里，虽然只声明了 `rand` 一个依赖，然而 Cargo 还是额外获取了 rand 所需的其他 crate，rand 依赖它们来正常工作。下载完成后，Rust 编译依赖，然后使用这些依赖编译项目。


#### 3.2 Cargo.lock 文件确保构建是可重现的

Cargo 有一个机制来确保任何人在任何时候重新构建代码，都会产生相同的结果：Cargo 只会使用你指定的依赖版本，除非你又手动指定了别的。Rust 在第一次运行 `cargo build` 时建立了 `Cargo.lock` 文件，可以在 guessing_game 目录找到它。

当第一次构建项目时，Cargo 计算出所有符合要求的依赖版本并写入 `Cargo.lock` 文件。当将来构建项目时，Cargo 会发现 `Cargo.lock` 已存在并使用其中指定的版本，而不是再次计算所有的版本。这使得能够拥有一个自动化的可重现的构建。


#### 3.3 更新 crate

当确实需要更新 crate 的时候，可以使用 `cargo update`命令，它会忽略 Cargo.lock 文件，并计算出所有符合 Cargo.toml 声明的最新版本。Cargo 接下来会把这些版本写入 Cargo.lock 文件。不过，Cargo 默认只会寻找大于或等于 0.8.3 而小于 0.9.0 的版本。如果需要 `rand` 使用 `0.9.0` 版本或任何 `0.9.x` 系列的版本，则必须像这样更新 `Cargo.toml` 文件：

```bash
[dependencies]
rand = "0.9.0"
```

下一次运行 `cargo build` 时，Cargo 会从 registry（注册源） 更新可用的 crate，并根据指定的新版本重新计算。


#### 3.4 生成一个随机数

修改 `src/main.rs`:

```rust
use std::io;
use rand::Rng;

fn main() {
    println!("Guess the number!");

    let secret_number = rand::thread_rng().gen_range(1..101);

    println!("The secret number is: {}", secret_number);

    println!("Please input your guess.");

    let mut guess = String::new();

    io::stdin()
        .read_line(&mut guess)
        .expect("Failed to read line");

    println!("You guessed: {}", guess);
}
```

首先，新增了一行 `use rand::Rng`。`Rng` 是一个 trait，它定义了随机数生成器应实现的方法，想使用这些方法的话，此 trait 必须在作用域中。

接下来，在中间添加两行。在首行中，调用 `rand::thread_rng` 函数来提供将要使用的特定随机数生成器：它位于当前执行线程的本地环境中，并从操作系统获取 seed。然后调用随机数生成器的 `gen_range` 方法。该方法由刚才使用 `use rand::Rng` 语句引入的 Rng trait 定义。gen_range 方法获得一个区间表达式（range expression）作为参数，并在区间内生成一个随机数。在这里使用的区间表达式采用的格式为 `start..end`。它包括起始端，但排除终止端（前闭后开）。所以需要指定 1..101 生成一个 1 到 100 之间的数字。或者可以传入区间 `1..=100`，这和前面的表达等价。

*注：运行 `cargo doc --open` 命令来构建所有本地依赖提供的文档，并在浏览器中打开。可以快速查看所需的 crate 使用说明文档，进一步知道应该 use 哪个 trait 以及该从 crate 中调用哪个方法。*


新添加的第二行代码打印出了秘密数字。

尝试运行程序几次：

```bash
Guess the number!
The secret number is: 78
Please input your guess.

Guess the number!
The secret number is: 91
Please input your guess.

Guess the number!
The secret number is: 10
Please input your guess.
```

每次可以得到不同的随机数，符合预期。



### 4 比较猜测的数字和秘密数字



`src/main.rs`:

```rust
use rand::Rng;
use std::cmp::Ordering;
use std::io;

fn main() {
    // --snip--
    println!("Guess the number!");

    let secret_number = rand::thread_rng().gen_range(1..101);

    println!("The secret number is: {}", secret_number);

    println!("Please input your guess.");

    let mut guess = String::new();

    io::stdin()
        .read_line(&mut guess)
        .expect("Failed to read line");

    println!("You guessed: {}", guess);

    match guess.cmp(&secret_number) {
        Ordering::Less => println!("Too small!"),
        Ordering::Greater => println!("Too big!"),
        Ordering::Equal => println!("You win!"),
    }
}
```

首先我们增加了另一个 use 声明，从标准库引入了一个叫做 `std::cmp::Ordering` 的类型到作用域中。`Ordering` 也是一个枚举，不过它的成员是 `Less`、`Greater` 和 `Equal`。这是比较两个值时可能出现的三种结果。

接着，底部的五行新代码使用了 `Ordering` 类型，`cmp` 方法用来比较两个值并可以在任何可比较的值上调用。它获取一个被比较值的引用：这里是把 guess 与 secret_number 做比较。 然后它会返回一个刚才通过 `use` 引入作用域的 `Ordering` 枚举的成员。使用一个 `match` 表达式，根据对 guess 和 secret_number 调用 cmp 返回的 Ordering 成员来决定接下来做什么。

一个 `match` 表达式由分支（arm） 构成。一个分支包含一个用于匹配的模式（pattern），给到 `match` 的值与分支模式相匹配时，应该执行对应分支的代码。Rust 获取提供给 `match` 的值并逐个检查每个分支的模式。模式和 match 结构是 Rust 中强大的功能，它体现了代码可能遇到的多种情形，并帮助你确保没有遗漏处理。

这里的代码编译会发生报错：

```bash
cargo build

error[E0308]: mismatched types
  --> src/main.rs:23:21
   |
23 |     match guess.cmp(&secret_number) {
   |                 --- ^^^^^^^^^^^^^^ expected `&String`, found `&{integer}`
   |                 |
   |                 arguments to this method are incorrect
   |
   = note: expected reference `&String`
              found reference `&{integer}`
note: method defined here
  --> /rustc/9fc6b43126469e3858e2fe86cafb4f0fd5068869/library/core/src/cmp.rs:964:8

For more information about this error, try `rustc --explain E0308`.
error: could not compile `guessing_game` (bin "guessing_game") due to 1 previous error
```

错误的核心表明这里有**不匹配的类型**（mismatched type）。Rust 有一个静态强类型系统，同时也有类型推断。当我们写出 `let guess = String::new()` 时，Rust 推断出 `guess` 应该是 `String` 类型，并不需要我们写出类型。另一方面，`secret_number` 是数字类型。Rust 中有好几种数字类型拥有 1 到 100 之间的值：`32 位数字 i32`、`32 位无符号数字 u32`、`64 位数字 i64`，等等。Rust 默认使用 i32，这是 `secret_number` 的类型，除非额外指定类型信息，或任何能让 Rust 推断出不同数值类型的信息。这里错误的原因在于 Rust 不会比较字符串类型和数字类型。

所以我们必须把从输入中读取到的 `String` 转换为一个真正的数字类型，才好与秘密数字进行比较。这可以通过在 main 函数体中增加如下两行代码来实现：

```rust
let guess: u32 = guess.trim().parse().expect("Please type a number!");
```

这里创建了一个叫做 `guess` 的变量。Rust 允许用一个新值来遮蔽 （shadow） `guess` 之前的值。这允许我们复用 guess 变量的名字，而不是被迫创建两个不同变量。

将这个新变量绑定到 `guess.trim().parse()` 表达式上。表达式中的 `guess` 是指原始的 `guess` 变量，其中包含作为字符串的输入。`String` 实例的 `trim` 方法会去除字符串开头和结尾的空白字符，我们必须执行此方法才能将字符串与 u32 比较，因为 u32 只能包含数值型数据。用户必须输入 `enter` 键才能让 `read_line` 返回，并输入他们的猜想，这会在字符串中增加一个换行符。例如，用户输入 5 并按下 `enter`，`guess` 看起来像这样：`5\n`。`trim` 方法会消除 `\n` 或 `\r\n`，只留下 5。

字符串的 `parse` 方法 将字符串解析成数字。因为这个方法可以解析多种数字类型，因此需要告诉 Rust 具体的数字类型，这里通过 `let guess: u32` 指定。`guess` 后面的冒号（`:`）告诉 Rust 我们指定了变量的类型。Rust 有一些内建的数字类型；`u32 是一个无符号的 32 位整型`。另外，程序中的 `u32` 标注以及与 `secret_number` 的比较，意味着 Rust 会推断出 `secret_number` 也是 `u32` 类型。现在可以使用相同类型比较两个值了！

由于 `parse` 方法只能用于可以逻辑转换为数字的字符，所以调用它很容易产生错误。因此，`parse` 方法返回一个 `Result` 类型，再次使用 `expect` 方法处理潜在的错误。如果 `parse` 不能从字符串生成一个数字，返回一个 `Result` 的 `Err` 成员时，`expect` 会使游戏崩溃并打印附带的信息。如果 `parse` 成功地将字符串转换为一个数字，它会返回 `Result` 的 `Ok` 成员，然后 `expect` 会返回 `Ok` 值中的数字。


再次运行程序：

```bash
cargo run

Guess the number!
The secret number is: 48
Please input your guess.
1
You guessed: 1
Too small!
```

现在代码的行为基本就符合预期了。在给代码加个循环，让用户一直猜，直到猜中为止。



### 5 使用循环来允许多次猜测

`loop` 关键字创建一个无限循环，让用户可以一直猜；增加一个`break`语句，在用户猜对时退出游戏。

```rust
use rand::Rng;
use std::cmp::Ordering;
use std::io;

fn main() {
    println!("Guess the number!");

    let secret_number = rand::thread_rng().gen_range(1..101);

    println!("The secret number is: {}", secret_number);

    loop {
        println!("Please input your guess.");

        let mut guess = String::new();

        io::stdin()
            .read_line(&mut guess)
            .expect("Failed to read line");

        let guess: u32 = guess.trim().parse().expect("Please type a number!");

        println!("You guessed: {}", guess);

        // --snip--

        match guess.cmp(&secret_number) {
            Ordering::Less => println!("Too small!"),
            Ordering::Greater => println!("Too big!"),
            Ordering::Equal => {
                println!("You win!");
                break;
            }
        }
    }
}

```

将提示用户猜测之后的所有内容放入了循环。确保 loop 循环中的代码多缩进四个空格。

通过在 `You win!` 之后增加一行 `break`，用户猜对了神秘数字后会退出循环。退出循环也意味着退出程序，因为循环是 `main` 的最后一部分。


#### 5.1 处理无效输入

不要在用户输入非数字时崩溃，需要忽略非数字，让用户可以继续猜测。可以通过修改 `guess` 将 `String` 转化为 `u32` 那部分代码来实现：

```rust
let guess: u32 = match guess.trim().parse() {
    Ok(num) => num,
    Err(_) => continue,
};
```

将 `expect` 调用换成 `match` 语句，从而实现遇到错误就崩溃转换成处理错误。如果 `parse` 能够成功的将字符串转换为一个数字，它会返回一个包含结果数字的 `Ok`。这个 `Ok` 值与 `match` 第一个分支的模式相匹配，该分支对应的动作返回 `Ok` 值中的数字 `num`，最后如愿变成新创建的 `guess` 变量。如果 `parse` 不能将字符串转换为一个数字，它会返回一个包含更多错误信息的 `Err`。`Err` 值不能匹配第一个 `match` 分支的 `Ok(num)` 模式，但是会匹配第二个分支的 `Err(_)` 模式：`_` 是一个通配符值，这里用来匹配所有 `Err` 值，不管其中有何种信息。程序会执行第二个分支的动作，`continue` 意味着进入 `loop` 的下一次循环，请求另一个猜测。


最后，再把打印 `secret_number`的语句删掉即可。

```rust
use rand::Rng;
use std::cmp::Ordering;
use std::io;

fn main() {
    println!("Guess the number!");

    let secret_number = rand::thread_rng().gen_range(1..101);

    loop {
        println!("Please input your guess.");

        let mut guess = String::new();

        io::stdin()
            .read_line(&mut guess)
            .expect("Failed to read line");

            let guess: u32 = match guess.trim().parse() {
                Ok(num) => num,
                Err(_) => continue,
            };

        println!("You guessed: {}", guess);

        // --snip--

        match guess.cmp(&secret_number) {
            Ordering::Less => println!("Too small!"),
            Ordering::Greater => println!("Too big!"),
            Ordering::Equal => {
                println!("You win!");
                break;
            }
        }
    }
}
```

玩一把试试：

```bash
cargo run
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 0.01s
     Running `target/debug/guessing_game`
Guess the number!
Please input your guess.
50
You guessed: 50
Too big!
Please input your guess.
25
You guessed: 25
Too small!
Please input your guess.
37
You guessed: 37
Too small!
Please input your guess.
43
You guessed: 43
Too small!
Please input your guess.
47
You guessed: 47
Too big!
Please input your guess.
45
You guessed: 45
You win!
```

