---
layout: post
title: textile_basic
date: 2023-12-23
tags: [tools]
author: taot
---

# textile 语法

## 1、文字修饰

修饰行内文字

| **字体样式** | **textile 语法** | **对应的 XHTML 语法** | **实际显示效果** |
|---|---|---|---|
|加强|`*strong*`|`<strong>strong</strong>`|<strong>strong</strong>|
|强调|`_emphasis_`|`<em>emphasis</em>`|<em>emphasis</em>|
|加粗|`**bold**`|`<b>bold</b>`|<b>bold</b>|
|斜体|`__italics__`|`<i>italics</i>`|<i>italics</i>|
|加大字号|`++bigger++`|`<big>bigger</big>`|<big>bigger</big>|
|减小字号|`--smaller--`|`<small>smaller</small>`|<small>smaller</small>|
|删除线（删除文字）|`-deleted text-`|`<del>deleted text</del>`|<del>deleted text</del>|
|下划线（插入文字）|`+inserted text+`|`<ins>inserted text</ins>`|<ins>inserted text</ins>|
|上标|`Text ^superscript^`|`Text <sup>superscript</sup>`|Text <sup>superscript</sup>|
|下标|`Text ~subscript~`|`Text <sub>subscript</sub>`|Text <sub>subscript</sub>|
|span|`%span%`|`<span>span</span>`|<span>span</span>|
|行内代码块|`code`|`<code>code</code>`|<code>code</code>|

## 2、段落修饰

修饰段落用于指定当前段落的属性

|**用途**|**textile语法**|**对应的 XHTML 语法**|
|---|---|---|
|普通段落|`p.`|`<p></p>`|
|标题|`hn.`|`<hn></hn>`|
|块状引用|`bq.`|`<blockquote></blockquote>`|
|块状代码|`bc.`|`<code></code>`|

### 1）普通段落 `p.`

一般情况下 `p.`` 这个修饰符可以忽略，因为所有的空行后面新起的一行会被 textile 解释器解释为普通段落

下面两种写法等价
```
a linea
a new linear
```

```
p. a line
p. a new line
```

都会被解释为 HTML:
```
<p>一行文字</p>
<p>空行后的一行文字</p>
```

### 2）标题文字 `hn.`

这里 `n` 表示 1-7 的数字 
```
h2. 二级标题
```

会被解释为 HTML：
```
<h2>二级标题</h2>
```


### 3）块引用 `bq.`

表示整段引用的文字
```
bq. 块状引用
```
会被解释为 HTML：
```
<blockquote>
<p>块状引用</p>
</blockquote>
```

### 4）代码块 `bc.`

使用 `bc.` 表示代码块
```
bc. 代码块
```

会被解释为：

```<pre>
<code>代码块</code>
</pre>
```

正常情况下，这些段落以一个新的换行符为结束的标志。但有时候一个段落内需要包含换行符（比如多行的代码），这时候需要将段落的定义延伸

## 3、延伸段落范围

一般来说，换行符开始会被认作段落的结束标志，但是如果要在一个段落包含两个以上的新行，只需要在修饰符（比如p.）后面再多加一个点，告诉textile 解析器，段落内有多个新行，直到出现另一个段落标记（p.）

```bc.. 本段落会一直延伸到一个 p. 结束。
这是第二段。
这是第三段。
p. 这一段不包括在上面的段落内。
```


## 4、转义符

有些字符不需要解释为Textile语言，这时就需要转义符。Textile语言的转义符是相连的两个等号"=="，用两个等号围起来的部分，就输出为原始的 HTML 格式，不做解析

```
An asterisk is used as in ==*.*== .
```

将输出 `An asterisk is used as in *.*. `

中间的句点就不会被被强调加粗


##. 5、列表

HTML 里面的常用的列表有两种：有序（ordered list, ol）和无序(unordered list, ul)，在textile里分别用 `#` 和 `*` 表示
```
* one
* two
* three
```

输出：
* one
* two
* three

也可以构造多级列表：
```
* one
** one A
** one B
*** one B1
* two
** two A
** two B
* three
```

输出：
* one
  * one A
  * one B
    * one B1
* two
  * two A
  * two B
* three

有序列表与之类似：
```
# one
## one A
## one B
### one B1
# two
## two A
## two B
# three
```

输出：
1. one
   1. one A
   2. one B
      1. one B1
2. two
   1. two A
   2. two B
3. three

## 6、内联修饰符

内联修饰符是指添加在已有 Textile 表达式中的进一步的修饰符，有以下几种：
```
{style rule}，花括号表示一个CSS的样式集，相当于"style= style rule",
[language]，中括号表示语言种类，相当于"lang=language",
(css class)，小括号表示CSS类，相当于"class=css class",
(#identifier)，小括号加井字号表示CSS的id，相当于"id=identifier".
```

这些内联符号不能单独修饰文字，必须与被修饰的HTML实体结合使用，比如`<span>、<p>`等

## 7、超链接

Textile 表示超链接的格式:
```
"显示文字":链接
```

如果要给链接添加 title，使用小括号:
```
"显示文字(title)":链接
```