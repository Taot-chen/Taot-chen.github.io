---
layout: post
title: git_branch_tag
date: 2024-04-29
tags: [git]
author: taot
---


## git branch与tag操作

### 1 git新建本地分支并推送到远程

新建本地分支，并切换到新分支上
```bash
git checkout -b 新分支名
```

新建一个远程分支，名字一样
```bash
git push origin 新分支名:新分支名
```

将本地分支和远程分支合并关联
```bash
git push --set-upstream origin 新分支名
```


### 2 使用Git添加Tag

#### 2.1 查看标签

**打印所有标签**
```bash
git tag
```

**打印符合检索条件的标签**
```bash
git tag -l <tag_name>
```
如 `git tag -l 1.*.*` 为搜索一级版本为1的版本

**查看对应标签状态**
```bash
git checkout <tag_name>
```

#### 2.2 创建本地标签

**创建轻量标签**，轻量标签指向一个发行版的分支，其只是一个像某commit的引用，不存储名称时间戳及标签说明等信息。定义方法如下
```bash
git tag <tag_name>-light
```

**创建带附注标签**，相对于轻量标签，附注标签是一个独立的标签对象，包含了名称时间戳以及标签备注等信息，同时指向对应的commit。定义方法如下
```bash
git tag -a <tag_name> -m "<备注信息>"
```

也可以向特定的commit添加标签，使用该commit对应的SHA值即可
```bash
git tag -a <tag_name> <SHA值> -m "<备注信息>"
```

**删除本地标签**
```bash
git tag -d <tag_name>
```

#### 2.3 将本地标签提交到远程仓库

**推送所有标签**
```bash
git push origin --tags
```

**推送指定版本的标签**
```bash
git push origin <tag_name>
```

#### 2.4 删除远程仓库的标签

同创建本地标签一样，删除了本地标签之后也要同时删除远程仓库的标签

**新版本Git (> v1.7.0)**
```bash
git push origin --delete <tag_name>
```

**新旧版本通用方法**，旧版本Git并没有提供直接删除的方法，可以通过将一个空标签替换现有标签来实现删除标签
```bash
git push origin :refs/tags/<tag_name>
```

### 3 git 重命名本地和远程分支

#### 3.1 git 删除本地和远程分支

在大多数情况下，删除 Git 分支很简单。

一个 Git 仓库常常有不同的分支，开发者可以在各个分支处理不同的特性，或者在不影响主代码库的情况下修复 bug。开发人员完成处理一个特性之后，常常会删除相应的分支。

* 删除本地分支

```bash
git branch -d <branch>
```

如果你还在一个分支上，那么 Git 是不允许你删除这个分支的。所以，请记得退出需要删除的分支：`git checkout master`。

当一个分支被推送并合并到远程分支后，-d 才会本地删除该分支。如果一个分支还没有被推送或者合并，那么可以使用-D强制删除它。

* 删除远程分支

```bash
git push <remote> --delete <branch>

# 有些版本的 git 无法使用上面的命令删除远程分支，那么可以使用下面的命令删除远程分支，这个方法通常是通用的
git push <remote> :<branch> # 例如 git push origin  :<branch>
```

最后，同步分支列表：
```bash
git fetch -p
```

#### 3.2 git 重命名本地和远程分支

* 重命名本地分支

在项目开发过程中，有时会需要重命名分支。当本地的开发分支还没有推送到远程分支的时候，可以在本地进行分支的重命名：

```bash
# 当处于需要重命名的分支时
git branch -m new_branch_name

# 当不在需要重命名的分支时
git branch -m old_branch_name new_branch_name
```

* 重命名远程分支

当分支已经推送至远端的时候，可以按照下面的步骤进行重命名。

```bash
# 重命名本地分支
git branch -m new_branch_name

# 删除远程分支
git push origin  :old_branch_name

# 上传新命名的本地分支
git push origin new_branch_name:new_branch_name

# 关联修改后的本地分支与远程分支
git push --set-upstream origin new_branch_name
```
