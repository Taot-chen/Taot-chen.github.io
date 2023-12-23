---
layout: post
title: tips_for_linux_command
date: 2023-12-23
---

# linux 命令行的一些小技巧

## 1. Create a tar archive split into blocks of a maximum size
在将文件打包并在服务间使用 ssh 协议传输的时候，单个归档文件过大往往会有较大传输失败的风险，可以在使用 tar 命令归档的时候，把归档拆分成多个小文件，在传输完成后再拼接回来
打包并拆分：
```bash
    tar cvzf - source_dir/ | split --bytes=1024MB - target_file.tar.gz.
```
拆分完成后会生成一系列大小为 1GB 的归档文件：target_file.tar.gz.aa，target_file.tar.gz.ab，target_file.tar.gz.ac, ...
拼接：
```bash
    cat target_file.tar.gz.* | tar xzvf -
```


## 2. reverse search history（平时开发中非常有用的一个高效快捷键）
* 反向查找/搜索历史命令
  * `ctrl+r` 从当前行开始向后搜索，并根据需要在历史记录中向上移动。这是一个增量搜索。此命令将区域设置为匹配的文本并激活标记
    * 按下 Ctrl + r 组合键，进入反向搜索状态
    * 输入查找字符串，显示历史命令中能够匹配成功的最近执行的一条命令
      * 为了提高查找效率，应该输入要查找命令中最特别的字符 (别的命令不包含的字符)，避免和其他执行过的相近命令更早匹配成功
      * 继续按下 Ctrl + r 组合键，可以继续向前搜索匹配命令
        * 多次按下 Ctrl + r 组合键，可以继续向前搜索匹配命令
    * 按下 -> 键，退出搜索状态/交互模式

