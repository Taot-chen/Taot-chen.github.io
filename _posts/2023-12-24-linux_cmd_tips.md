---
layout: post
title: linux_cmd_tips
date: 2023-12-24
tags: [tools]
author: taot
---

### Tips of Terminal

### 一、文件复制

#### 1、将多个文件同时复制到另一个位置

```shell
# 将文件file1,file2,file3 从 /home/dir1/ 复制到 /home/dir2/
cp /home/dir1/{file1,file2,file3} -r /home/dir2/
# file1,file2,...可以是文件或者文件夹，也可以既有文件又有文件夹
# 参数 -r 选用：如果file1,file2,...中包含文件夹，就需要参数 -r ，否则就不需要
```

#### 2、将一个文件同时复制到多个位置

```shell
# 将文件 file 从 /home/dir1/ 复制到 /home/dir2/ 和 /home/dir3
echo /home/dir2/ /home/dir3/ | xargs -n 1 cp -n /home/dir1/file -r
# 命令解释：
# echo 命令将输出打印到屏幕上，通过 | 讲、将输出重定向到 xargs 命令。xargs 命令从  echo 命令获得两次输入，并执行两次 cp 命令，将文件复制到两个指定的位置
# xargs 命令的参数 -n 1 告诉 cp 命令一次接受一个参数
# cp 命令的 -n 参数可以让目标位置中若已经存在相同文件名的文件，则跳过本次复制，提高效率
```

#### 3、将多个文件同时复制到多个位置

```shell
# 将文件 file1,file2,file3 从 /home/dir1/ 复制到 /home/dir2/ 和 /home/dir3
echo /home/dir2/ /home/dir3/ | xargs -n 1 cp -n /home/dir1/{file1,file2,file3} -r
```

### 二、`xargs`命令

**`xargs`命令：** 可以将管道或者标准输入（stdin）的数据转换成命令行参数，也能够从文件的输出中读取参数，一般是和管道符一起使用

```shell
命令格式：
	#l 命令 | xargs 选项 选项的值
		选项说明：
			-a filename：从文件中读入作为stdin，如：xargs -a 1.txt（就是读取1.txt的内容作为下一参数的stdin）
			-E flag：flag必须是一个以空格分隔的标志，当xargs分析到含有flag这个标志的时候就停止
			
			-p：交互式打印。当每次执行一个argument的时候询问一次用户
			-t:表示先打印命令，然后再执行
			-n num：后面加次数，表示命令在执行的时候一次性用的argument的个数，默认是用所有
			
			-i或-I：将xargs的每项名称，一般是一行一行赋值给{}，可以用{}代替
			-r no-run-if-empty：当xargs的输入为空的时候则停止xargs，不用再去执行
			-d delim：指定分割符，默认的xargs分割符是回车，argument的分隔符是空格。


演示：
		[root@server ~]# cat 1.txt
			a
			b
			c
			d
			1234
			2
			3

	# xargs -a 1.txt		=>	-a 读取文件内容
		[root@server ~]# xargs -a 1.txt
			a b c d 1234


	# xargs -a 1.txt -E c		=> 通过-E 指定一个标志，让xargs执行到这个标志就停止
		[root@server ~]# xargs -a 1.txt -E c
			a b
		[root@server ~]# xargs -a 1.txt -E 1		=> 因为没有1这个标志，所以xargs不会执行-E 选项
			a b c d 1234 2 3
		[root@server ~]# xargs -a 1.txt -E 2		=>	有2这个标志，会停止执行
			a b c d 1234

	# xargs -a 1.txt -p			=> -p 会询问是否打印，并且会告诉你 -a 选项是如何打印出来的，通过echo命令
	[root@server ~]# xargs -a 1.txt -p
		echo a b c d 1234 2 3 ?...y			=>	输入 y 执行打印
		a b c d 1234 2 3
	[root@server ~]# xargs -a 1.txt -p
		echo a b c d 1234 2 3 ?...n			=>	输入 n 不执行打印

	# xargs -a 1.txt -t				=>	-t 选项和-p选项一样，但是不会询问，直接打印出来
		[root@server ~]# xargs -a 1.txt -t
		echo a b c d 1234 2 3
		a b c d 1234 2 3

	# xargs -a 1.txt -n1			=> 通过制定 -n 选项指定一行输出几个数据
		[root@server ~]# xargs -a 1.txt -n1
			a
			b
			c
			d
			1234
			2
			3
	[root@server ~]# xargs -a 1.txt -n2
			a b
			c d
			1234 2
			3
		[root@server ~]# xargs -a 1.txt -n3
			a b c
			d 1234 2
			3

	# xargs -a 1.txt -d "-"			=>	-d 选项改掉列与列的默认分隔符为其他，就会换行
		[root@server ~]# xargs -a 1.txt -d "-"
			a
			b
			c
			d
			1234
			2
			3
		[root@server ~]# xargs -a 1.txt		=>	不修改默认分隔符是空回车，全部一行打印
			a b c d 1234 2 3

```



