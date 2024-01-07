---
layout: post
title: model_deployment_4_modify_tune_onnx
date: 2024-01-06
tags: [onnx]
author: taot
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

在用 API 对 ONNX 模型进行操作之前，我们还需要先了解一下 ONNX 的结构定义规则，学习一下 ONNX 在 Protobuf 定义文件里是怎样描述一个神经网络的。
回想一下，神经网络本质上是一个计算图。计算图的节点是算子，边是参与运算的张量。而通过可视化 ONNX 模型，我们知道 ONNX 记录了所有算子节点的属性信息，并把参与运算的张量信息存储在算子节点的输入输出信息中。事实上，ONNX 模型的结构可以用类图大致表示如下：

![onnx结构定义](../blog_images/github_drawing_board_for_gitpages_blog/onnx结构定义.png)

如图所示，一个 ONNX 模型可以用 ModelProto 类表示。ModelProto 包含了版本、创建者等日志信息，还包含了存储计算图结构的 graph。GraphProto 类则由输入张量信息、输出张量信息、节点信息组成。张量信息 ValueInfoProto 类包括张量名、基本数据类型、形状。节点信息 NodeProto 类包含了算子名、算子输入张量名、算子输出张量名。
让我们来看一个具体的例子。假如我们有一个描述 output=a*x+b 的 ONNX 模型 model，用 print(model) 可以输出以下内容：

```python
ir_version: 8 
graph { 
  node { 
    input: "a" 
    input: "x" 
    output: "c" 
    op_type: "Mul" 
  } 
  node { 
    input: "c" 
    input: "b" 
    output: "output" 
    op_type: "Add" 
  } 
  name: "linear_func" 
  input { 
    name: "a" 
    type { 
      tensor_type { 
        elem_type: 1 
        shape { 
          dim {dim_value: 10} 
          dim {dim_value: 10} 
        } 
      } 
    } 
  } 
  input { 
    name: "x" 
    type { 
      tensor_type { 
        elem_type: 1 
        shape { 
          dim {dim_value: 10} 
          dim {dim_value: 10} 
        } 
      } 
    } 
  } 
  input { 
    name: "b" 
    type { 
      tensor_type { 
        elem_type: 1 
        shape { 
          dim {dim_value: 10} 
          dim {dim_value: 10} 
        } 
      } 
    } 
  } 
  output { 
    name: "output" 
    type { 
      tensor_type { 
        elem_type: 1 
        shape { 
          dim { dim_value: 10} 
          dim { dim_value: 10} 
        } 
      } 
    } 
  } 
} 
opset_import {version: 15} 
```

对应上文中的类图，这个模型的信息由 ir_version，opset_import 等全局信息和 graph 图信息组成。而 graph 包含一个乘法节点、一个加法节点、三个输入张量 a, x, b 以及一个输出张量 output。在下一节里，我们会用 API 构造出这个模型，并输出这段结果。


### 2 读写 onnx 模型

#### 2.1 构造 onnx 模型
