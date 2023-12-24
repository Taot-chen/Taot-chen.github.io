---
layout: post
title: markdown_basic
date: 2023-12-24
---

## 一、标题

&emsp;在想要设置为标题的文字前加#
&emsp;一个#表示一级标题，两个#表示二级标题，**支持六级标题。**
&emsp;注：标准语法要求#和标题之间有空格，但是现在大部分版本有没有空格都可以。
<br/>
&emsp; **_示例：_** <br>

# 这是一级标题

## 这是二级标题

#### 这是四级标题

##### 这是五级标题

###### 这是六级标题

---

## 二、字体

- **加粗**<br>
    &emsp;要加粗的文字左右分别用两个\*号包起来
    <br/>

- **斜体**<br>
    &emsp;要倾斜的文字左右分别用一个\*号包起来
    <br/>

- **斜体加粗**<br>
    &emsp;要倾斜和加粗的文字左右分别用三个\*号包起来
    <br/>

- **删除线**<br>
    &emsp;要加删除线的文字左右分别用两个~~号包起来
    <br/>

- **字体大小**<br>
    &emsp;`<font size=num>`设置需要的字号`</font>`
    <br/>

- **颜色**<br>
    &emsp;`<font color=color_style>`设置需要的文字`</font>`颜色
    <br/>

- **字体类型**<br>
    &emsp;`<font face=“字体名字”>`设置字体类型`</font>`
    <br/>

&emsp; **_示例：_** <br>
**这是加粗的文字**<br>
_这是倾斜的文字_<br>
**_这是斜体加粗的文字_**<br>
~~这是加删除线的文字~~<br>
<font size=5> hello</font><br>
<font color="blue"> <font size=4> 4 号 bule</font></font><br>
<font face="微软雅黑"> <font color=pink> 粉色微软雅黑</font></font><br>

---

## 三、引用

在引用的文字前加 **>** 即可
引用可以嵌套：
<：一级引用
<<：二级引用
n 个
...
貌似可以一直嵌套
</br>
&emsp; **_示例：_** <br>

> 这是引用的内容
>
> > 这是引用的内容
> >
> > > > > 这是引用的内容

---

## 四、分割线

&emsp; 三个或者三个以上 **-** 或者 **\***
&emsp; **_示例：_** <br>

---

---

---

***

## 五、图片

&emsp;语法：<br>
`![图片alt](图片地址 "图片title")`

> 图片 alt 就是显示在图片下面的文字，相当于对图片内容的解释。
> 图片 title 是图片的标题，当鼠标移到图片上时显示的内容。title 可加可不加

**为保证多平台统一，图片的存储需要使用图床，提供统一的外链。如此才能做到书写一次，各处使用。**
<br/>

&emsp; **_示例：_** <br>
![blockchain]("/home/taot/Notes/6860761-fd2f51090a890873.jpg", "区块链")<br>

---

## 六、超链接

&emsp;语法：<br>
`[超链接名](超链接地址 "超链接title")`

> title 可加可不加

<br>

&emsp; **_示例：_** <br>
[百度](http://baidu.com)<br>
[谷歌](http://google.com "谷歌")<br>
<br/>

---

## 七、列表

#### 无序列表

&emsp;语法：<br>
无序列表使用 **+** 、**-** 、 **\*** 任何一个后跟列表项内容。<br>
&emsp; **_示例：_** <br>

+ 列表内容<br>

* 列表内容<br>

- 列表内容<br>

> `注意：- + \* 跟内容之间都要有一个空格`

<br/>

#### 有序列表

&emsp; 语法：<br>
`数字+.`<br>
&emsp; **_示例：_** <br>

1. 列表内容<br>
2. 列表内容<br>
3. 列表内容<br>
    3.1.   列表内容

> `注意：序号跟内容之间要有空格`

<br/>

####列表嵌套
**上一级和下一级之间加三个空格即可** <br>

---

## 八、表格

&emsp; 语法：<br>

```
| 表头 | 表头  | 表头 |
| ---- | :---: | ---: |
| 内容 | 内容  | 内容 |
| 内容 | 内容  | 内容 |
```

> 第二行分割表头和内容。
>
> - 有一个就行，为了对齐，多加了几个
>     文字默认居左 -两边加：表示文字居中 -右边加：表示文字居右
>     注：原生的语法两边都要用 | 包起来。此处省略

<br>

&emsp; **_示例：_** <br>

| 姓名 | 技能 | 排行 |
| ---- | :--: | ---: |
| 刘备 |  哭  | 大哥 |
| 关羽 |  打  | 二哥 |
| 张飞 |  骂  | 三弟 |
| <br> |      |      |

---

## 九、代码

&emsp; 语法：<br>
单行代码两端使用反引号括起来（反引号是英语输入法时 esc 键下面那个键）<br>
&emsp; **_示例：_** <br>
`code`<br>
<br/>
代码块：代码之间分别用三个反引号包起来，且两边的反引号单独占一行<br>
&emsp; **_示例：_** <br>

```
  代码...
  代码...
  代码...
```

<br/>

---

## 十、流程图

&emsp; 语法：<br>

````
    ```flow
    st=>start: 开始
    op=>operation: My Operation
    cond=>condition: Yes or No?
    e=>end
    st->op->cond
    cond(yes)->e
    cond(no)->op
    ```
````

<br>

&emsp; **_示例：_** <br>

```flow
    st=>start: 开始
    op=>operation: My Operation
    cond=>condition: Yes or No?
    e=>end
    st->op->cond
    cond(yes)->e
    cond(no)->op
```

---

## 十一、补充

**markdown 支持 HTML 标签，所以可以使用内嵌 HTML 来实现这些功能。**<br>
<br/>

- **制表符**<br>
    &emsp; 语法：`&emsp;`<br>
    &emsp; **_示例：_** <br>
    &emsp; 制表符测试
    这是一段关于在 markdown 里面使用制表符的测试
    <br/>

- **空行**<br>
    &emsp; HTML 换行标签：`<br/>`
    &emsp; 在需要换行的任意的位置输入`<br/>`可以实现换行效果，输入几个就会换几行。
    <br/>

- **字体大小**<br>
    &emsp;`<font size=num>`设置需要的字号`</font>`<br>
    &emsp; **_示例：_** <br>
    <font size=5> hello</font><br>
    <br/>

- **颜色**<br>
    &emsp;`<font color=color_style>`设置需要的文字`</font>`颜色<br>
    &emsp; **_示例：_** <br>
    <font color="blue"> <font size=4> 4 号 bule</font></font><br>
    <br/>

- **字体类型**<br>
    &emsp;`<font face=“字体名字”>`设置字体类型`</font>`<br>
    &emsp;**_示例：_** <br>
    <font face="微软雅黑"> <font color=pink> 粉色微软雅黑</font></font><br>
    <br/>

- **换行**<br>
    &emsp;在需要换行的地方使用`<br>`
    <br/>