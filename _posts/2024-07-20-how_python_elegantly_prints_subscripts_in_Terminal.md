---
layout: post
title: how_python_elegantly_prints_subscripts_in_Terminal
date: 2024-07-20
tags: [python]
author: taot
---

## Python如何优雅地在Terminal打印下标

在 Python 中想要再terminal窗口打印下表，可以使用`Unicode`方法将下标打印到 terminal 窗口也可以使用`\N{}`转义序列将下标打印到 terminal 窗口。

### 1 使用`Unicode`方法将下标打印到 terminal 窗口

在 Python 中，没有直接的方法可以将下标打印到 terminal 窗口。我们需要参考这个[Unicode subscripts and superscripts](https://en.wikipedia.org/wiki/Unicode_subscripts_and_superscripts)来查看我们想要放在下标或上标符号中的字符的 Unicode 表示。

然后我们在我们的`print()`函数中使用`\u`转义字符编写该表示，以告诉解释器我们正在使用 Unicode 表示法编写。

数字的 Unicode 表示以 `\u208` 开头，后跟所需的数字，字母的 Unicode 表示以 `\u209` 开头，后跟该字母的索引。

```python
print(u"H\u2082SO\u2084")
print("x\u2091")

# 输出
H₂SO₄
xₑ
```

在第一行用\u2082 和\u2084 打印下标数字；和第二行的下标字母与\u2091。在第二行中，1 是序列的最后一个字符，而我们得到 e 作为下标。这是因为 e 在此 Unicode 表示中具有索引 1。

### 2 使用`\N{}`转义序列将下标打印到 terminal 窗口

记住每个字符和符号的`unicode`索引可能会比较困难，我们可以使用这种方法来缓解我们的困难并使代码更具可读性。我们可以使用要打印的符号的别名来进行打印。[Superscripts and Subscripts](https://www.unicode.org/charts/nameslist/n_2070.html)为 Unicode 中的所有别名提供了方便的指南。

我们可以在`\N{}`转义序列中写入这些别名，以将它们打印到控制台。

```python
print("CO\N{subscript two}")
print("C\N{LATIN SUBSCRIPT SMALL LETTER N}")

# 输出：
CO₂
Cₙ
```

此 Unicode 名称不区分大小写，这意味着我们可以使用大写或小写的 Unicode 名称。
