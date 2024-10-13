---
layout: post
title: python_versions
date: 2024-10-12
tags: [python]
author: taot
---

## python各版本新特性

python 语言经历了从 python2 到 pyhton3 的大版本更新，python3 也有了十几次比较大的版本更新，每次更新都有自己的新特性和改进，这里整理一下每次版本更新都有些什么大的变化。

### 1 python 2

Python 2 是 Python 的一个老版本，它的最后一个大版本是 Python 2.7。这个版本在2010年发布，它的新特性包括：
* 改进的输入函数；
* 关键字参数和默认参数；
* 对象迭代器；
* 简化的类定义语法。

### 2 Python 3

Python 3 是 Python 的最新版本，在 2008 年发布。Python 3 不兼容 Python 2，但它引入了许多新特性。

#### 2.1 Python 3.0

Python 3.0 是 Python 3 系列的首个版本，于 2008 年发布。该版本引入了许多重要的变化，其中一些是为了解决 Python 2 中存在的设计缺陷和不一致性。Python 3.0 的一些主要特性：
* **print 函数**：Python 3.0 中，print 语句变成了一个函数，使用了新的语法。例如，`print "Hello, World!"`在 Python 3.0中变成了`print("Hello, World!")`。
* **整数除法**：在Python 3.0中，整数除法的结果将总是得到浮点数，即使被除数和除数都是整数。
* **Unicode支持**：Python 3.0采用了更加一致和统一的Unicode支持，字符串默认为Unicode字符串。

#### 2.2 Python 3.1

Python 3.1于2009年发布，该版本主要集中在性能改进和bug修复上。Python 3.1的一些主要特性：
* **垃圾回收**：Python 3.1引入了改进的垃圾回收机制，提高了内存管理的效率。
* **多线程**：Python 3.1中的多线程支持得到了改进，提供了更好的线程安全性和性能。
* **新的库和模块**：Python 3.1引入了一些新的标准库和模块，包括**unittest模块的改进和添加了fractions模块等**。

#### 2.3 Python 3.2

Python 3.2于2011年发布，该版本引入了一些新的特性和改进，包括：

* **concurrent.futures模块**：Python 3.2引入了concurrent.futures模块，提供了一个高级的接口来处理并发任务。
* **yield from语法**：Python 3.2中，引入了yield from语法，简化了使用生成器的代码。
* **functools.lru_cache装饰器**：Python 3.2引入了functools.lru_cache装饰器，提供了一个简单而有效的缓存机制。

#### 2.4 Python 3.3

Python 3.3于2012年发布，该版本引入了一些新的语言特性和库改进，包括：

* **yield表达式**：Python 3.3中，yield语句可以作为表达式使用，可以将值发送给生成器。
* **venv模块**：Python 3.3引入了venv模块，用于创建和管理虚拟环境。
* **新的语法特性**：Python 3.3引入了一些新的语法特性，如yield from语句，以及更好的异常链式处理。

#### 2.5 Python 3.4

Python 3.4于2014年发布，该版本引入了一些新的语言特性和库改进，包括：

* **asyncio库**：Python 3.4引入了asyncio库，提供了一种基于协程的异步编程模型。
* **enum模块**：Python 3.4引入了enum模块，用于定义枚举类型。
* **pathlib模块**：Python 3.4引入了pathlib模块，提供了一种更简洁和面向对象的路径操作API调用错误。

#### 2.6 Python 3.5

Python 3.5于2015年发布，该版本引入了一些新的语言特性和库改进，包括：

* **async/await语法**：Python 3.5引入了async/await语法，使得异步编程更加简洁和易于理解。
* **类型提示**：Python 3.5开始支持类型提示，通过给函数和变量添加类型注解，可以提供更好的代码可读性和静态类型检查。
* **新的标准库模块**：Python 3.5引入了一些新的标准库模块，如**typing模块用于类型提示，以及zipapp模块用于创建可执行的ZIP应用**。

#### 2.7 Python 3.6

Python 3.6于2016年发布，该版本引入了许多新的语言特性和改进，包括：

* **字典排序**：Python 3.6中，字典保持插入顺序，使得字典的迭代顺序可预测。
* **f-strings**：Python 3.6引入了f-strings，一种新的字符串格式化语法，提供了更简洁和直观的方式来格式化字符串。
* **异常链式处理**：Python 3.6支持异常链式处理，可以在异常处理中显式地关联多个异常。

#### 2.8 Python 3.7

Python 3.7于2018年发布，该版本引入了一些新的语言特性和库改进，包括：

* **数据类**：Python 3.7引入了数据类，通过使用简单的语法，可以自动为类生成一些常见的方法，如init和repr。
* **异步生成器**：Python 3.7中，引入了异步生成器语法，用于更方便地处理异步编程中的迭代器。
* **上下文变量绑定**：Python 3.7引入了上下文变量绑定语法，使得在with语句中可以将上下文管理器的结果绑定到一个变量。

#### 2.9 Python 3.8

Python 3.8于2019年发布，该版本引入了一些新的语言特性和库改进，包括：

* **Walrus运算符**：Python 3.8引入了Walrus运算符（:=），允许在表达式中进行变量赋值。
* **f-strings改进**：Python 3.8对f-strings进行了改进，支持在格式化字符串中使用等号和括号。
* **异步迭代器和异步生成器改进**：Python 3.8对异步迭代器和异步生成器进行了改进，提供了更好的语法和性能。

#### 2.10 Python 3.9

Python 3.9于2020年发布，该版本引入了一些新的语言特性和库改进，包括：

* **字典合并运算符**：Python 3.9引入了字典合并运算符（|），用于合并两个字典。
* **类型提示改进**：Python 3.9对类型提示进行了改进，支持更多的类型注解语法和类型推断。
* **新的标准库模块**：Python 3.9引入了一些新的标准库模块，如**zoneinfo模块用于处理时区信息，以及graphlib模块用于处理图形数据结构**。

#### 2.11 Python 3.10

Python 3.10于2021年发布，该版本引入了一些新的语言特性和库改进，包括：

* **匹配模式**：Python 3.10引入了匹配模式（match statement），它是一种更简洁和直观的模式匹配语法，可以用于替代复杂的if-elif-else结构。
* **结构化的异常上下文**：Python 3.10对异常上下文进行了改进，使得异常的上下文信息更加结构化和易于访问。
* **zoneinfo模块改进**：Python 3.10对zoneinfo模块进行了改进，提供了更好的时区支持和操作。

#### 2.12 Python 3.11 

Python 3.10于2022年发布：

* **引入了一些新的语法和语言特性**，例如：**Structural Pattern Matching**、**compatible parentheses**，提升了代码的可读性和可维护性。
* **新增了装饰符（decorator）的语法糖**，这个改进使得装饰符在使用的时候更加方便。
* **像匿名函数（lambda）一样定义函数类型注释**，这有助于开发者更好地阐明函数的使用场景和语法。同时也支持了在函数暂定时指定返回类型。
* **使用 PEP 634 简化删除了装饰符函数之后 Python 的语法组成**，现在的语法是 `func := [decorators] "def" funcname "(" parameter_list ")" ["->" expression] ":" suite`。
* **引入了运行中的类型注释**：**typing.Annotated**，它允许程序员将元数据嵌入他们的函数参数、返回值、变量等，从而为这些变量/参数加上额外的类型注释和其他详细文档。

#### 2.13 Python 3.12

Python 3.12于2023年发布：

* 新的语法特性
  * 类型形参语法和 type 语句
  * f-字符串 语法改进
* 解释器改进
  * 单独的每解释器 GIL
  * 低开销的监控
  * 针对 NameError, ImportError 和 SyntaxError 异常 改进了 'Did you mean ...' 提示消息
* 对 Python 数据模型的改进
  * 使用 Python 的 缓冲区协议
* 标准库中的重大改进
  * pathlib.Path 类现在支持子类化
  * os 模块获得了多项针对 Windows 支持的改进
  * 在 sqlite3 模块中添加了 命令行界面。
  * 基于 运行时可检测协议 的 isinstance() 检测获得了 2 至 20 倍的提速
  * asyncio 包的性能获得了多项改进，一些基准测试显示有 75% 的提速。
  * 在 uuid 模块中添加了 命令行界面。
  * 通过 tokenize 模块生成令牌（token）的速度最多可提高 64%
* C API 的改进：
  * 不稳定 C API 层
  * 永生对象
* CPython 实现的改进：
  * 推导式内联化
  * 对 Linux perf 性能分析器的 CPython 支持
  * 在受支持的平台上实现栈溢出保护
* 新的类型标注特性：
  * 使用 TypedDict 来标注 **kwargs
  * typing.override() 装饰器
* 重要的弃用、移除或限制：
  * 在 Python 的 C API 中移除 Unicode 对象中的 wstr，使每个 str 对象的大小缩减至少 8 个字节。
  * 移除 distutils 包。
  * 不在使用 venv 创建的虚拟环境中预装 setuptools。 这意味着 distutils、setuptools、pkg_resources 和 easy_install 默认将不再可用；要访问这些工具请在 激活的 虚拟环境中运行 pip install setuptools。
  * 移除了 asynchat、asyncore 和 imp 模块，以及一些 unittest.TestCase 方法别名。

#### 2.14 Python 3.13

Python 3.13于2024年发布：

* 解释器的改进：
  * 大幅改进的 交互式解释器 和 改进的错误消息。
  * 现在 locals() 内置函数在修改被返回的映射时具有 已定义语义。 Python 调试器及类似的工具现在即使在并发代码执行期间也能更可靠地在已优化的作用域中更新局部变量。
  * CPython 3.13 具有对在运行时禁用 global interpreter lock 的实验性支持。
  * 增加了一个基本的 JIT 编译器。 目前默认是禁用的（但以后可能启用）。 能够小幅提升性能
  * 在新的 交互式解释器 中，以及 回溯信息 和 文档测试 输出中的颜色支持。 这可以通过 PYTHON_COLORS and NO_COLOR 环境变量来禁用。
* 对 Python 数据模型的改进：
  * __static_attributes__ 保存了可在一个类体的任何函数中通过 self.X 来访问的属性名称。
  * __firstlineno__ 记录了一个类定义的首行的行号。
* 标准库中的重大改进：
  * 新增了 PythonFinalizationError 异常，当操作在 最终化 期间被阻塞时将被引发。
  * 现在 argparse 模块可支持弃用命令行选项、位置参数和子命令。
  * 新增的函数 base64.z85encode() 和 base64.z85decode() 支持对 Z85 数据 进行编码和解码。
  * 现在 copy 模块有一个 copy.replace() 函数，支持许多内置类型和任何定义了 __replace__() 方法的类。
  * 新的 dbm.sqlite3 模块现在是默认的 dbm 后端。
  * os 模块增加了 一套新函数 用于处理 Linux 的定时器通知文件描述符。
  * 现在 random 模块提供了一个 命令行界面。
* 安全改进：
  * ssl.create_default_context() 设置了 ssl.VERIFY_X509_PARTIAL_CHAIN 和 ssl.VERIFY_X509_STRICT 作为默认的 flag。
* C API 的改进：
  * 现在 Py_mod_gil 槽位被用来指明一个扩展模块支持在禁用 GIL 的情况下运行。
  * 增加了 PyTime C API，提供了对系统时钟的访问。
  * PyMutex 是新增的轻量级互斥锁，只占用一个字节。
  * 新增了 一套函数 用于在 C API 中生成监控事件。
* 新的类型标注特性：
  * 类型形参 (typing.TypeVar, typing.ParamSpec 和 typing.TypeVarTuple) 现在可支持默认值。
  * 新的 warnings.deprecated() 装饰器在类型系统和运行时中增加了对标记为弃用的支持。
  * typing.ReadOnly 可被用来将 typing.TypedDict 的项标记为对类型检查器只读。
  * typing.TypeIs 提供了更直观的类型细化行为，作为对 typing.TypeGuard 的替代。
* 平台支持：
  * 现在 Apple 的 iOS 是 官方支持的平台，处于 第 3 层级。
  * 现在 Android 是 官方支持的平台，处于 第 3 层级。
  * 现在 wasm32-wasi 作为 第 2 层级 的平台受到支持。
  * wasm32-emscripten 不再是受到官方支持的平台。
* 重要的移除：
  * 剩余的 19 个老旧 stdlib 模块已从标准库中移除: aifc, audioop, cgi, cgitb, chunk, crypt, imghdr, mailcap, msilib, nis, nntplib, ossaudiodev, pipes, sndhdr, spwd, sunau, telnetlib, uu 和 xdrlib。
  * 移除了 2to3 工具和 lib2to3 模块（在 Python 3.11 中已被弃用）。
  * 移除了 tkinter.tix 模块（在 Python 3.6 中已被弃用）。
  * 移除了 locale.resetlocale() 函数。
  * 移除了 typing.io 和 typing.re 命名空间。
  * 移除了链式的 classmethod 描述器。
