---
layout: post
title: github_drawing_board_for_gitpages_blog
date: 2024-01-05
---

## 借助 github 仓库来作为 gitpages 博客图床

在平时写文档的时候，尤其是在类似Markdown这种纯文本的文档中，使用图片常常很麻烦。

如果使用网络上的图片，可能出现以后链接不可用的状况；或者把图片上传到网络上再使用图片链接，往往图片链接千奇百怪，并且图片分散在网络上，不方便管理；使用云对象存储来自己搭建图床，还需要考虑对象存储续费或者几年后图床搬家的问题。

借助 github 来作为图床仓库，生成图片的引用链接。

### 1 使用 github 仓库做图片库生成图片引用链接

对于 gitpages 仓库 https://github.com/username/username.github.io，会被默认作为 Github Pages 的内容来源。上传到 username.github.io 的内容，都可以通过 https://username.github.io/ 访问到，可以通过目录进行不同类型、来源和作用的图片区分。

使用链接：https://username.github.io/完整目录/完整文件名.文件后缀 就可以引用到图片。图片可以通过git版本管理工具进行上传，也可以在github的网页上上传。这样就可以把图片和文档分离开，文档可以更精简；并且图片还可以通过版本管理工具进行统一管理。

这种方式也是我比较倾向的方式，平时也使用比较多。一般的流程：
* 在 markdown 编辑器设置把添加 markdown 的图片在本地放进指定路径，这样 markdown 中就会自动添加图片的本地路径链接；
* 根据 markdown 中的图片本地路径链接，把他们通过 git push 到 https://github.com/username/username.github.io 仓库的指定文件夹内；
* 把 markdown 中的图片本地路径链接替换成 https://github.com/username/username.github.io 仓库中对应的图片链接
* 更新 markdown 文件到 https://github.com/username/username.github.io 仓库的 _post 文件夹
*上面的第二、三步都可以通过脚本自动完成，实际用起来和其他的付费图床体验差不离。*


*另外，还有一种比较方便的方式，在本地 markdown 编辑器设置好，添加图片的时候，使用相对路径，这样在 push 的时候，就直接 push 所有 markdown 文件和 图片文件的改动即可*

