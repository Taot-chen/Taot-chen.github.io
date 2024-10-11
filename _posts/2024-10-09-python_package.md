---
layout: post
title: python_package
date: 2024-10-09
tags: [python]
author: taot
---

## python发包

Python 中我们经常会用到第三方的包，默认情况下，用到的第三方工具包基本都是从 [Pypi.org](https://pypi.org/) 里面下载。这些第三方的包都是开发者们发布的自己的库。我们有自己的想法，或者有一些常用的方法想要分享出去，就可以发布自己的库，也就是我们常说的造轮子。

PyPI (Python Package Index) 是 python 官方的第三方库的仓库，所有人都可以下载第三方库或上传自己开发的库到PyPI。PyPI 推荐使用 pip 包管理器来下载第三方库。截至目前，PyPI 已经有 574,662 个项目，很多知名项目都发布在上面。

造轮子的步骤：
* 包源代码开发
* git 版本管理
* 编写`setup.py`
* 编写说明文档
* 发布到 Pypi
* 后续维护升级


### 1 包源代码开发

包的功能可以使各种各样的，关于包的源代码编写就不过多阐述，这里就只是写了一个件简单的例子，用来输出`Hello Pypi`。项目工程地址：https://github.com/Taot-chen/hellopypi/tree/main

#### 1.1 创建项目必须文件

```bash
touch README.md hellopypi.py setup.py
```
文件结构：
```bash
hellopypi/
├── hellopypi.py
├── README.md
└── setup.py
```

#### 1.2 创建 git 仓库

现在我们已经创建了项目结构，下面将初始化一个 GitHub 存储库来托管代码：
```bash
git init
git add *
git commit -m "init repo"
git branch -M main
git remote set-url origin https://<your_token>github.com/<USERNAME>/hellopypi.git
git push -u origin main
```

也可以通过在 github 手动建好仓库之后，再通过 clone 建好的仓库，之后再往仓库添加文件的方式。

#### 1.3 包源代码开发

这里的主程序就是前面的`hellopypi.py`，里面的内容很简单：
```python
__version__ = '0.1.0'

def hello_pypi():
    print("Hello Pypi!")

def main():
  hello_pypi()

if __name__ == '__main__':
    main()
```

#### 1.4 编写`setup.py`

`setup.py`是每个能从 PyPi 上能下载到的库都有的文件，它是发布的关键所在。

kennethreitz 大神编写了一个 for human 的`setup.py`模板，项目地址：[传送门](https://github.com/kennethreitz/setup.py/blob/master/setup.py)，只需要把它复制过来，修改自己项目需要的地方即可，不需要额外的编写`setup.cfg`等其他文件。

我这里修改完的内容如下：
```python
#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Note: To use the 'upload' functionality of this file, you must:
#   $ pipenv install twine --dev

import io
import os
import sys
from shutil import rmtree

from setuptools import find_packages, setup, Command

# Package meta-data.
NAME = 'hellopypi'
DESCRIPTION = 'print Hello Pypi in terminal.'
URL = 'https://github.com/Taot-chen/hellopypi'
EMAIL = 'oehuosi@foxmail.com'
AUTHOR = 'oehuosi'
REQUIRES_PYTHON = '>=3.6.0'
VERSION = '0.1.0'

# What packages are required for this module to be executed?
REQUIRED = [
    # 'requests', 'maya', 'records',
]

# What packages are optional?
EXTRAS = {
    # 'fancy feature': ['django'],
}

# The rest you shouldn't have to touch too much :)
# ------------------------------------------------
# Except, perhaps the License and Trove Classifiers!
# If you do change the License, remember to change the Trove Classifier for that!

here = os.path.abspath(os.path.dirname(__file__))

# Import the README and use it as the long-description.
# Note: this will only work if 'README.md' is present in your MANIFEST.in file!
try:
    with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

# Load the package's __version__.py module as a dictionary.
about = {}
if not VERSION:
    project_slug = NAME.lower().replace("-", "_").replace(" ", "_")
    with open(os.path.join(here, project_slug, '__version__.py')) as f:
        exec(f.read(), about)
else:
    about['__version__'] = VERSION


class UploadCommand(Command):
    """Support setup.py upload."""

    description = 'Build and publish the package.'
    user_options = []

    @staticmethod
    def status(s):
        """Prints things in bold."""
        print('\033[1m{0}\033[0m'.format(s))

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        try:
            self.status('Removing previous builds…')
            rmtree(os.path.join(here, 'dist'))
        except OSError:
            pass

        self.status('Building Source and Wheel (universal) distribution…')
        os.system('{0} setup.py sdist bdist_wheel --universal'.format(sys.executable))

        self.status('Uploading the package to PyPI via Twine…')
        os.system('twine upload dist/*')

        self.status('Pushing git tags…')
        os.system('git tag v{0}'.format(about['__version__']))
        os.system('git push --tags')

        sys.exit()


# Where the magic happens:
setup(
    name=NAME,
    version=about['__version__'],
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type='text/markdown',
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    # packages=find_packages(exclude=["tests", "*.tests", "*.tests.*", "tests.*"]),
    # If your package is a single module, use this instead of 'packages':
    py_modules=['hellopypi'],

    entry_points={
        'console_scripts': ['hellopypi=hello:main'], 
    },
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    include_package_data=True,
    license='Apache-2.0',
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        'License :: OSI Approved :: Apache-2.0 license',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: Python :: Implementation :: PyPy'
    ],
    # $ setup.py publish support.
    cmdclass={
        'upload': UploadCommand,
    },
)
```

配置信息说明：

* 项目的配置信息：
```python
# Package meta-data.
NAME = 'mypackage'
DESCRIPTION = '填写你的项目简短描述.'
URL = 'https://github.com/你的github账户/mypackage'
EMAIL = 'me@example.com'    # 你的邮箱
AUTHOR = 'Awesome Soul'     # 你的名字
REQUIRES_PYTHON = '>=3.6.0' # 项目支持的python版本
VERSION = '0.1.0'           # 项目版本号
```

* 项目的依赖库(没有就不填)：
```python
# What packages are required for this module to be executed?
REQUIRED = [
    # 'requests', 'maya', 'records',
]
```

* setup部分:

这里大部分内容都不用填，只有以下几个注意点
  * `long_description`这里默认是项目的`README.md`文件
  * 注释掉的`entry_points`部分是用来生成命令行工具或者GUI工具的（理论上是跨平台的），这里我生成了一个`hellopypi`的命令来代替`hello.py`的`main`函数，安装成功以后就可以直接使用`hellopypi`命令：
    `entry_points={ 'console_scripts': ['hellopypi=hello:main'], },`
  * 如果你的项目文件夹下只有一个`py`文件来实现你的功能的话，需要将`packages=find_packages(exclude=["tests", "*.tests", "*.tests.*", "tests.*"])`,注释掉，然后取消`py_modules`的注释并进行相应修改。
```python
setup(
    name=NAME,
    version=about['__version__'],
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type='text/markdown',
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    # packages=find_packages(exclude=["tests", "*.tests", "*.tests.*", "tests.*"]),
    # If your package is a single module, use this instead of 'packages':
    py_modules=['hellopypi'],

    entry_points={
        'console_scripts': ['hellopypi=hello:main'], 
    },
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    include_package_data=True,
    license='Apache-2.0',
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        'License :: OSI Approved :: Apache-2.0 license',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: Python :: Implementation :: PyPy'
    ],
    # $ setup.py publish support.
    cmdclass={
        'upload': UploadCommand,
    },
)
```


#### 1.5 编写说明文档

一个好的项目，需要有一个条理清晰的文档的，在 README.md 对项目进行详尽的说明。

### 2 发布到 Pypi

#### 2.1 生成分发档案

为包生成分发包。这些是上传到包索引的档案，可以通过pip安装。

* 确保有`setuptools`, `wheel` 安装了最新版本：
```bash
python3 -m pip install --user --upgrade setuptools wheel
```

* 检查`setup.py`是否有错误:
  运行`python setup.py check`，如果没报错误，则输出一般是`running check`；如果有错误，就根据报错信息来修一下。

准备好上面的步骤, 一个包就基本完整了, 剩下的就是打包了。

##### 2.1.1 生成 tar.gz 包

```bash
python3 setup.py sdist build
```

在当前目录的 dist 文件夹下, 就会多出一个`tar.gz`结尾的包了。

##### 2.1.2 也可以打包一个 wheel 格式的包

```bash
python3 setup.py bdist_wheel --universal
```

在 dist 文件夹下面生成一个`whl`文件.

也可以一次性生成`tar.gz`包和`whl`包：
```bash
python3 setup.py sdist bdist_wheel
```

会在dist目录下生成一个`tar.gz`的源码包和一个`.whl`的 Wheel 包。

#### 2.2 发布包到 Pypi

先去[pypi](https://pypi.org/account/register/)注册账号，记住账号和密码，后面上传包会使用。



注册号账号之后，接下来就是上传包。

上传的时候会用到`twine`，需要先安装`twine`(用 twine上传分发包，并且只有 twine> = 1.11.0 才能将元数据正确发送到 Pypi上)。
```bash
pip install twine
```
前面编写的`setup.py`具备上传包的功能：
```bash
python3 setup.py upload
```

不出意外的话，到这里，我们自己的包就发布完成了。但是这里可能会遇到这样的报错：`The user 'xxx' isn't allowed to upload to project 'xxx'. See https://pypi.org/help/#project-name for more information.`

这个是由于**软件包名字是PyPI用以区分的唯一标识，因此必须全球唯一**，此时表明可能已经存在了相同名字的包了，那么换个不重复的名字即可。我这里就遇到了这个问题，因此我把名字改成了`hellopypi_oh`就可以了。



#### 2.3 验证发布 PYPI 成功

上传完成了会显示 success, 我们直接可以在 PyPI 上看到。

可以使用`pip`来安装包并验证它是否有效:
```bash
pip install hellopypi_oh
```

安装成功之后，直接在 Terminal 中执行 `hellopypi_oh`命令，看到输出
```bash
Hello Pypi!
```

表明发布成功。

### 3 后续维护升级

* 有更新升级之后，首先删除旧版本打包文件，然后生成新文件：
```bash
python3 setup.py sdist bdist_wheel
```

* 输入以下命令，上传新版本即可：
```bash
python setup.py upload
```
这个命令还会自动把代码改动更新到 github 仓库。
