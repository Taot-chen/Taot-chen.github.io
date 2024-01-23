---
layout: post
title: python_build_agent_pool
date: 2024-01-06
tags: [python]
author: taot
---

## python 创建代理池

爬虫程序是批量获取互联网上的信息的重要工具，在访问目标网站时需要频繁发送请求，为了避免被目标网站封禁 IP 地址，我们需要使用代理 IP 来代替自己的 IP 地址进行访问。此时，就需要用到代理池。

### 1 代理 IP & 代理池

代理IP是指由第三方提供的，可用于代替用户本机IP地址的IP地址。在网络爬虫或其他数据爬取场景中，使用代理IP可以实现以下几个目的：
* 防止 IP 被封禁：有些网站为了防止被爬虫攻击，会设置 IP 访问频率限制和封禁机制，使用代理 IP 可以规避这种封禁。
* 隐藏本机 IP：使用代理 IP 可以隐藏用户本机 IP，从而保护用户真实身份在互联网上的安全性。
* 改变访问区域：使用代理 IP 可以模拟其他地区或国家的 IP 地址，从而达到一定的访问效果。

代理 IP 池是指一个程序管理的，由多个代理 IP 组成的 IP 地址池。通常，代理 IP 池由两部分组成：一个是获取代理 IP 的部分，另一个是维护代理 IP 的部分。

获取代理 IP 的部分主要实现从各种渠道或代理商购买、租用、抓取到代理 IP，并将其存储在一个地址池中。

维护代理 IP 的部分主要实现对代理 IP 的筛选、检测、评分和淘汰等等。通过对代理 IP 进行维护，可以保证代理 IP 池中的 IP 地址有效可靠，避免无效的 IP 地址被使用。

### 2 构建代理池

Python 中实现代理 IP 池主要有以下几个步骤：
* 从网上获取代理 IP 地址，构建 IP 地址池。
* 对 IP 地址进行筛选，保留可用的 IP 地址。
* 使用筛选出来的 IP 地址进行数据的爬取。
* 对爬取过程中返回结果进行处理，过滤掉无用的数据。
* 在爬取过程中检测代理 IP 的可用性，将不可用的 IP 地址从 IP 地址池中删除。

#### 2.1 从网上获取代理 IP 地址

爬取代理 IP 的方法有多种，其中比较常用的有爬取代理网站、从代理商处购买或租用代理 IP、从代理池中抓取代理 IP 等等。

在此处，我们通过从代理网站上爬取代理 IP，并将其存储在代理 IP 池中的方法实现代理 IP 池的构建。

```python
import requests
from lxml import etree
import random

class ProxyPool:
    # 初始化代理池
    def __init__(self):
        # 从代理网站上获取代理 IP
        self.proxy_urls = [
            'http://www.zdaye.com/free/',
            'http://www.zdaye.com/free/2',
            'http://www.zdaye.com/free/3',
            'http://www.zdaye.com/free/4',
            'http://www.zdaye.com/free/5',
        ]
        self.proxies = self.get_proxies()

    # 获取代理 IP
    def get_proxies(self):
        proxies = []
        for url in self.proxy_urls:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'}
            response = requests.get(url, headers=headers)
            html = etree.HTML(response.text)
            ips = html.xpath('//table[@id="ip_list"]/tr/td[2]/text()')
            ports = html.xpath('//table[@id="ip_list"]/tr/td[3]/text()')
            for i in range(len(ips)):
                proxies.append('http://' + ips[i] + ':' + ports[i])
        return self.check_proxies(proxies)

    # 检查代理 IP 的可用性
    def check_proxies(self, proxies):
        valid_proxies = []
        for proxy in proxies:
            try:
                requests.get('https://www.baidu.com', proxies={'http': proxy, 'https': proxy}, timeout=3)
                valid_proxies.append(proxy)
            except:
                continue
        return valid_proxies

    # 随机获取代理 IP
    def get_proxy(self):
        proxy = random.choice(self.proxies)
        return proxy

```

#### 2.2 对 IP 地址进行筛选

为了保证代理 IP 稳定可用，我们需要定期对代理 IP 进行筛选和检测，将不可用的 IP 地址从 IP 地址池中删除。

```python
import time

class ProxyPool:
    ...
    # 定时检查代理 IP 的可用性
    def check_valid_proxies(self):
        while True:
            valid_proxies = self.check_proxies(self.proxies)
            self.proxies = valid_proxies
            time.sleep(60 * 60)

if __name__ == '__main__':
    proxy_pool = ProxyPool()
    proxy_pool.check_valid_proxies()

```

#### 2.3 使用筛选出来的 IP 地址进行数据的爬取

使用筛选出来的 IP 地址进行数据爬取时，需要注意以下几点：
* 每个 IP 地址的使用时间不宜过长，建议使用后及时更换。
* 使用 IP 地址时不要过于频繁，否则容易被封禁。
* 针对不同的网站需根据情况设置不同的请求头部和请求参数。
* 在爬取过程中检测代理 IP 的可用性，将不可用的 IP 地址从 IP 地址池中删除。

```python
class Spider:
    # 爬取目标网页
    def get_html(self):
        try:
            proxy = proxy_pool.get_proxy()
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'}
            response = requests.get(url, headers=headers, proxies={'http': proxy, 'https': proxy}, timeout=5)
            response.encoding = 'utf-8'
            html = response.text
            return html
        except:
            return None

if __name__ == '__main__':
    proxy_pool = ProxyPool()
    spider = Spider()
    while True:
        html = spider.get_html()
        if html is not None:
            # 对返回结果进行处理，过滤掉无用的数据
            ...
        else:
            print('IP地址失效，更换中...')

```

> 技术不分好坏，好坏在于使用技术的人。      -- 沃 · 兹基硕德
