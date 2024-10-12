---
layout: post
title: linux_libstdc++_so_6_version_GLIBCXX_3_4_30_not_found
date: 2024-07-15
tags: [linux]
author: taot
---

## linux解决报错`libstdc++.so.6: version 'GLIBCXX_3.4.30' not found`

最近使用的机器有多人操作，gcc 被其他人使用`apt-get`更新过了，导致需要使用 gcc 进行编译的工程在编译过程中出现`libstdc++.so.6: version 'GLIBCXX_3.4.30' not found`的报错。这个工程之前是可以正常编译的，猜测是更新之后，相应的文件丢失了或者是文件(动态链接指向的文件)出问题。

* 根据报错信息查看`libstdc++.so.6`文件的详细信息：
```bash
ls -al /path/to/libstdc++.so.6  # 这里 /path/to/libstdc++.so.6 在报错信息中会给出
```
应该可以看到是一个软连接，那么就和我们前面的猜测一致了，是动态链接指向的文件出问题了。

* 利用`strings`命令看一下指向的文件其GLIBCXX的东西是否能对应上(由于在编译工程的时候已经出现了报错，那么必然是对应不上的了，也就是查询的结果为空)
```bash
strings libstdc++.so.6 | grep GLIBCXX_3.4.30
# 不会有任何输出
```

* 从系统其他地方找符合要求的`libstdc++.so.6`，可以使用`locate`命令快速查找：
```bash
locate libstdc++.so.6
# 应该会出来一堆的结果
```

* 从`locate`命令的一堆输出中，找到`/usr/lib/x86_64-linux-gnu`这一条，并使用`string`命令看一下其GLIBCXX的东西
```bash
strings /usr/lib/x86_64-linux-gnu/libstdc++.so.6 | grep GLIBCXX_3.4.30

# 输出内容：GLIBCXX_3.4.30
```

* 重新创建软链接`/path/to/libstdc++.so.6`，时其指向`/usr/lib/x86_64-linux-gnu/libstdc++.so.6`
```bash
rm /path/to/libstdc++.so.6
ln -s /usr/lib/x86_64-linux-gnu/libstdc++.so.6 /path/to/libstdc++.so.6
```

不出意外的话，到这里前面的编译报错就不会出现了。

*在多用户的机器上，在需要使用和系统自带的版本不同的软件的时候，可以把需要的版本的软件安装到自己账户的`home`目录下，再通过设置环境变量了指定软件的位置即可，这样既不影响其他用户，也不会被其他用户影响。*
———— 沃·兹基硕德

