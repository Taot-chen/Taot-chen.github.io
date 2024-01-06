---
layout: post
title: model_deployment_4_modify_tune_onnx
date: 2024-01-06
---

## 修改和调试 onnx 模型

### 1. onnx 底层实现原理

#### 1.1 onnx 的存储格式

ONNX 在底层是用 Protobuf 定义的。*Protobuf，全称 **Protocol Buffer**，是 Google 提出的一套表示和序列化数据的机制。使用 Protobuf 时，用户需要先写一份数据定义文件，再根据这份定义文件把数据存储进一份二进制文件。数据定义文件相当于是数据类，二进制文件相当于是数据类的实例。*

例如，有这样一个 Protobuf 数据定义文件：
```cpp
    message Person { 
    required string name = 1; 
    required int32 id = 2; 
    optional string email = 3; 
} 
```
这段定义表示在 Person 这种数据类型中，必须包含 name、id 这两个字段，选择性包含 email字段。

根据这份定义文件，用户可以选择一种编程语言，定义一个含有成员变量 name、id、email 的 Person 类，把这个类的某个实例用 Protobuf 存储成二进制文件；同时，用户也可以用二进制文件和对应的数据定义文件，读取出一个 Person 类的实例。

对于 ONNX ，Protobuf 的数据定义文件在其开源库，这些文件定义了神经网络中模型、节点、张量的数据类型规范；二进制文件就是 `.onnx`文件，每一个 onnx 文件按照数据定义规范，存储了一个神经网络的所有相关数据。

直接用 Protobuf 生成 ONNX 模型比较麻烦，ONNX 提供了很多实用 API，可以在完全不了解 Protobuf 的前提下，构造和读取 ONNX 模型。

#### 1.2 onnx 结构定义

![onnx结构定义](../blog_images/github_drawing_board_for_gitpages_blog/onnx结构定义.png)

