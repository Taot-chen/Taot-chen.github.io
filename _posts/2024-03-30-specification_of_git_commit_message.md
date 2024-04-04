---
layout: post
title: specification_of_git_commit_message
date: 2024-03-30
tags: [git]
author: taot
---


## git commit message 规范

### 1 Git提交描述格式规范解析

一个规范的Git提交描述格式如下，包含**Header， Body，Footer**

```bash
# Header
[<type>](<scope>): <subject>

# Body
<body>

# Footer
<footer>
```

#### 1.1 Header

Header头只有一行,包括3个字段: **type(必需), scope(可选), subject(必需)**

|属性|描述|
|---|---|
|type(必填)|commit提交类型|
|scope(选填)|commint提交影响范围|
|subject(必填)|commint提交简短描述|

##### 1.1.1 type 提交类型

type说明提交类型：只允许使用下面属性

|属性|描述|
|---|---|
|feat|新功能|
|fix|修改bug|
|docs|文档修改|
|style|格式修改|
|refactor|重构|
|perf|性能提升|
|test|测试|
|build|构建系统|
|ci|对CI配置文件修改|
|chore|修改构建流程、或者增加依赖库、工具|
|revert|回滚版本|

##### 1.1.2 scope 作用范围

scope说明提交影响范围：一般是修改的什么模块或者是什么功能，如【xx模块】/【xx功能】

##### 1.1.3 subject 提交主题

subject 说明提交简短描述：一般是5-10个字简单描述做的任务，如【xx模块加入消息队列】

#### 1.2 Body

body说明提交详细描述：对于功能详细的描述，解释为什么加入这段代码，为什么调整优化等，如因分布式锁问题，导致死锁问题，优化调整xxxx

##### 1.3 Footer

Footer脚包括2个字段: **Breaking Changes、Closed Issues**

|属性|描述|
|---|---|
|Breaking Changes|中断性不兼容变动(不常用)|
|Closed Issues|关闭Issues问题|

##### 1.3.1 Breaking Changes

当前版本与之前版本不兼容，如迭代升级对之前版本不能做到兼容，就需要在Breaking Changes后面描述变动理由和迁移方法之类，此属性不常用

##### 1.3.2 Closed Issues

当前 commit提交针对某个issue问题或者是bug编号等，如Closes  # 234


### 2 git commit 模板

修改 ~/.gitconfig, 添加:
```bash
[commit]
template = ~/.gitmessage
```

新建 ~/.gitmessage 内容可以如下:
```bash
# head: [<type>](<scope>): <subject>
# - type: feat, fix, docs, style, refactor, test, chore, perf, build, ci, revert
# - scope: can be empty (eg. if the change is a global or difficult to assign to a single component)
# - subject: start with verb (such as 'change'), 50-character line
#
# body: 72-character wrapped. This should answer:
# * Why was this change necessary?
# * How does it address the problem?
# * Are there any side effects?
#
# footer: 
# - Include a link to the ticket, if any.
# - BREAKING CHANGE
#
```

在有新的 commit 的时候，在 `git add files` 之后，`git commit` 会打开 commit 模板，在模板上修改即可。

也可以进一步配置 commit message 检查等。
