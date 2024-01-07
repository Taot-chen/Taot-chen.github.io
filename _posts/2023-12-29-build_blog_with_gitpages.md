---
layout: post
title: build_blog_with_gitpages
date: 2023-12-29
tags: [tools]
author: taot
---

# build blog with gitpages

## 1 介绍

  博客整体效果。在线预览我的博客：https://taot-chen.github.io

  * 支持特性
    * 简约风格博客
    * Powered By Jekyll
    * 博客文章搜索
    * 自定义社交链接
    * 网站访客统计
    * Google Analytics 网站分析
    * Gitalk评论功能
    * 自定义关于about页面
    * 支持中文布局
    * 支持归档与标签

## 2 新建博客 git 仓库
  * 首先你要在 github 上新建自己博客仓库，用来生成和存放博客文章。你可以直接 fork/clone 我的博客仓库。这样你马上有了自己的博客仓库。在新建仓库的时候，需要把仓库的名称设置为固定的格式：`username.github.io`
    其中 username 是你的 github 用户名，github page解析的时候找的是这个 username.github.io的仓库名
    **=版权声明：fork/clone 之后 _posts 文件夹内容是我的博客文章，版权归我所有。你可以选择删除里面的文章替换上自己的或者转载附上链接注明出处**

    此时，不出意外的话，打开域名https://username.github.io 就能看到你刚搭建的博客了

  * 这个时候也可能会出现 404 页面，或者是渲染失败的源码页面。这种情况可能是 Jekyll build 失败造成的。实际上每次仓库有更新之后，Jekyll 都会重新 build 整个项目，当仓库更新有不恰当的内容（例如，语法错误）时，Jekyll 就会 build 失败，导致博客页面渲染失败。此时，只需要查看 Jekyll 的 build 日志，找到报错的地方 fix 并更新仓库即可

## 3 博客配置
  * 仓库根目录下的 _config.yml 文件是博客配置文件，根据自己的信息进行修改并更新到仓库即可
  * 博客名称和描述
    ```css
        # Name of your site (displayed in the header)
        name: "taot's blog"
        # Short bio or description (displayed in the header)
        description: "分享编程资源 | 学习路线 | 记录学习历程"
    ```
  * 社交链接
    这里配置社交链接按钮，没配的不显示
    ```css
        # Includes an icon in the footer for each username you enter
        footer-links:
          #weibo: frommidworld #请输入你的微博个性域名 https://www.weibo.com/<thispart>
          # behance: # https://www.behance.net/<username>
          dribbble:
          # zhihu: ning-meng-cheng-31-94
          email: 1624024615@qq.com
          facebook:
          flickr:
          github: Taot-chen
          googleplus: # anything in your profile username that comes after plus.google.com/
          instagram:
          linkedin:
          pinterest:
          rss: # just type anything here for a working RSS icon
          stackoverflow: # your stackoverflow profile, e.g. "users/50476/bart-kiers"
          tumblr: # https://<username>.tumblr.com
          #twitter: frommidworld
          youtube:
    ```
  * 配置gitalk
    这个是评论功能的配置。评论功能基于gitalk，在配置文件中找到gitalk配置项目，修改规则如下：
    ```css
        gitalk:
        clientID: <你的clientID>
        clientSecret: <你的clientSecret>
        repo: <你的repository名称>
        owner: <你的GitHub用户名>
    ```
    原理是利用github的issues评论文章
  * Google站长统计
    使用谷歌分析账号，它可以统计你博客网站的访问人数，访问来源等非常丰富的网站数据
    ```css
        # Enter your Google Analytics web tracking code (e.g. UA-2110908-2) to activate tracking
        google_analytics: UA-XXXXXXX-X
    ```
  * 博客网址配置
    ```css
        # Your website URL (e.g. http://barryclark.github.io or http://www.barryclark.co)
        # Used for Sitemap.xml and your RSS feed
        url: https://yourname.github.io
    ```
    done! 不出意外的话，现在访问上面提到的博客地址，就可以看到自己的博客。

## 4 如何写博客
  * 文章用 markdown 语法，写好统一放在 _post 文件夹下上传，git page 会自动从 github 仓库拉取并重新 build Jekyll，解析成网页，之后就能在博客网页浏览。
  * 关于文章的命名格式：博客文章必须按照统一的命名格式 yyyy-mm-dd-blogName.md，并且文章开头必须有统一的 head，例如本文的文件名为 `2023-12-29-build_blog_with_gitpages.md`，文件开头内容为：
    ```markdown
        ---
        layout: post
        title: build_blog_with_gitpages
        date: 2023-12-29
        ---
    ```

