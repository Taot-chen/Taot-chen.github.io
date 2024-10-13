---
layout: post
title: torch_source_code_3_torch_fx
date: 2024-07-26
tags: [torch]
author: taot
---


### 0 概述

FX 是一个供开发者用来转换 nn.Module 实例的工具包。FX 包含三个主要组件：**符号跟踪器（symbolic_traced）**、**中间表示（intermediate representation，IR）**和**Python 代码生成（Code generation）**。这些组件的应用实例：

```python
import torch
# Simple module for demonstration
class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.param = torch.nn.Parameter(torch.rand(3, 4))
        self.linear = torch.nn.Linear(4, 5)

    def forward(self, x):
        return self.linear(x + self.param).clamp(min=0.0, max=1.0)

module = MyModule()

from torch.fx import symbolic_trace
# Symbolic tracing frontend - captures the semantics of the module
symbolic_traced : torch.fx.GraphModule = symbolic_trace(module)

# High-level intermediate representation (IR) - Graph representation
print(symbolic_traced.graph)
"""
graph():
    %x : [num_users=1] = placeholder[target=x]
    %param : [num_users=1] = get_attr[target=param]
    %add : [num_users=1] = call_function[target=operator.add](args = (%x, %param), kwargs = {})
    %linear : [num_users=1] = call_module[target=linear](args = (%add,), kwargs = {})
    %clamp : [num_users=1] = call_method[target=clamp](args = (%linear,), kwargs = {min: 0.0, max: 1.0})
    return clamp
"""

# Code generation - valid Python code
print(symbolic_traced.code)
"""
def forward(self, x):
    param = self.param
    add = x + param;  x = param = None
    linear = self.linear(add);  add = None
    clamp = linear.clamp(min = 0.0, max = 1.0);  linear = None
    return clamp
"""
```

**符号跟踪器**对 Python 代码执行**符号执行**。它将称为代理的虚假值馈送到代码中。记录对这些代理的操作。

**中间表示**是符号跟踪期间记录的操作的容器。它包含一个节点列表，这些节点表示函数输入、调用点（到函数、方法或 torch.nn.Module 实例）以及返回值。

**Python 代码生成**是使 FX 成为 Python 到 Python（或模块到模块）转换工具包的关键。对于每个 Graph IR，我们可以创建与 Graph 语义匹配的 Python 代码。此功能封装在 GraphModule 中，它是一个 torch.nn.Module 实例，它包含一个 Graph 以及从 Graph 生成的 forward 方法。

总而言之，这个组件管道（**符号跟踪 -> 中间表示 -> 转换 -> Python 代码生成**）构成了 FX 的 Python 到 Python 转换管道。此外，这些组件可以单独使用。例如，符号跟踪可以单独使用来捕获代码的一种形式以进行分析（而不是转换）目的。代码生成可用于以编程方式生成模型，例如从配置文件生成模型。

#### 0.1 torch.fx.symbolic_trace(root, concrete_args=None)

torch.fx.symbolic_trace(root, concrete_args=None)[SOURCE](https://pytorch.org/docs/stable/_modules/torch/fx/_symbolic_trace.html#symbolic_trace)

* 参数
  * root (Union[torch.nn.Module, Callable])，要跟踪并转换为图形表示的模块或函数。
  * concrete_args (Optional[Dict[str, any]]) – 要部分特化的输入

* 返回值
  * 从 root 中记录的操作创建的模块。

* 返回类型
  * GraphModule

这是一个符号跟踪 API，给定一个 `nn.Module` 或函数实例 root，此函数将返回一个 `GraphModule`，该模块通过跟踪 root 中看到的操作来构建。`concrete_args` 允许对函数进行部分特化，无论是为了移除控制流还是数据结构。

例如有如下存在控制流的代码，
```python
def f(a, b):
    if b == True:
        return a
    else:
        return a*2
```

由于存在控制流，FX 通常无法跟踪此代码。但是，我们可以使用 concrete_args 对 b 的值进行特化，以跟踪此代码
```python
f = fx.symbolic_trace(f, concrete_args={'b': False})
assert f(3, False)  == 6
```
此时，仍然可以传入不同的 b 值，但它们将被忽略。

还可以使用 `concrete_args` 从函数中消除数据结构处理。这将使用 pytrees 来扁平化输入。为了避免过度特化，不应特化的值传入 fx.PH。例如
```python
def f(x):
    out = 0
    for v in x.values():
        out += v
    return out
f = fx.symbolic_trace(f, concrete_args={'x': {'a': fx.PH, 'b': fx.PH, 'c': fx.PH}})
assert f({'a': 1, 'b': 2, 'c': 4}) == 7
```

#### 0.2 class torch.fx.Tracer(autowrap_modules=(math,), autowrap_functions=())

class torch.fx.Tracer(autowrap_modules=(math,), autowrap_functions=())[SOURCE](https://pytorch.ac.cn/docs/stable/_modules/torch/fx/_symbolic_trace.html#Tracer)

Tracer 是实现 `torch.fx.symbolic_trace` 符号跟踪功能的类。调用 `symbolic_trace(m)` 等同于 `Tracer().trace(m)`。

可以对 Tracer 进行子类化，以覆盖跟踪过程的各种行为。


####  0.3 class torch.fx.Graph(owning_module=None, tracer_cls=None, tracer_extras=None)

class torch.fx.Graph(owning_module=None, tracer_cls=None, tracer_extras=None)[SOURCE](https://pytorch.ac.cn/docs/stable/_modules/torch/fx/graph.html#Graph)

Graph 是 FX 中间表示中使用的主要数据结构。它由一系列 Node 组成，每个节点代表调用站点（或其他语法结构）。Node 的列表共同构成一个有效的 Python 函数。

例如，以下代码
```python
import torch
import torch.fx

class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.param = torch.nn.Parameter(torch.rand(3, 4))
        self.linear = torch.nn.Linear(4, 5)

    def forward(self, x):
        return torch.topk(torch.sum(self.linear(x + self.linear.weight).relu(), dim=-1), 3)

m = MyModule()
gm = torch.fx.symbolic_trace(m)
```

将生成以下 Graph
```python
print(gm.graph)

graph(x):
    %linear_weight : [num_users=1] = self.linear.weight
    %add_1 : [num_users=1] = call_function[target=operator.add](args = (%x, %linear_weight), kwargs = {})
    %linear_1 : [num_users=1] = call_module[target=linear](args = (%add_1,), kwargs = {})
    %relu_1 : [num_users=1] = call_method[target=relu](args = (%linear_1,), kwargs = {})
    %sum_1 : [num_users=1] = call_function[target=torch.sum](args = (%relu_1,), kwargs = {dim: -1})
    %topk_1 : [num_users=1] = call_function[target=torch.topk](args = (%sum_1, 3), kwargs = {})
    return topk_1
```

##### 0.3.1 __init__(owning_module=None, tracer_cls=None, tracer_extras=None)

__init__(owning_module=None, tracer_cls=None, tracer_extras=None) [SOURCE](https://pytorch.ac.cn/docs/stable/_modules/torch/fx/graph.html#Graph.__init__)

构造一个空的图。

##### 0.3.2 call_function(the_function, args=None, kwargs=None, type_expr=None)

call_function(the_function, args=None, kwargs=None, type_expr=None) [SOURCE](https://pytorch.ac.cn/docs/stable/_modules/torch/fx/graph.html#Graph.call_function)

在 Graph 中插入一个 call_function Node。一个 call_function 节点表示对由 the_function 指定的 Python 可调用对象的调用。

* 参数
  * **the_function** (Callable[..., Any])，要调用的函数。可以是任何 PyTorch 运算符、Python 函数或 builtins 或 operator 命名空间的成员。
  * **args** (Optional[Tuple[Argument, ...]])，传递给被调用函数的位置参数。
  * **kwargs** (可选[Dict[str, Argument]])，传递给被调用函数的关键字参数
  * **type_expr** (可选[Any])，一个可选的类型注解，表示此节点输出的 Python 类型。
* 返回值，新创建并插入的 call_function 节点。
* 返回类型，节点

*此方法与`Graph.create_node()`具有相同的插入点和类型表达式规则。*

##### 0.3.3 call_method(method_name, args=None, kwargs=None, type_expr=None)

call_method(method_name, args=None, kwargs=None, type_expr=None) [SOURCE](https://pytorch.ac.cn/docs/stable/_modules/torch/fx/graph.html#Graph.call_method)

在 Graph 中插入一个 call_method Node。一个 call_method 节点表示对 args 中第 0 个元素的给定方法的调用。

* 参数
  * **method_name** (str)，要应用于 self 参数的方法名称。例如，如果 args[0] 是一个表示 Tensor 的 Node，那么要调用该 Tensor 上的 relu()，请将 relu 传递给 method_name。
  * **args** (可选[元组[参数, ...]])，传递给调用方法的位置参数。请注意，这应该包含一个 self 参数。
  * **kwargs** (可选[字典[字符串, 参数]])，传递给调用方法的关键字参数
  * **type_expr** (可选[Any])，一个可选的类型注解，表示此节点输出的 Python 类型。

* 返回值，新创建并插入的 call_method 节点。

* 返回类型，节点

*此方法与`Graph.create_node()`具有相同的插入点和类型表达式规则。*

##### 0.3.4 call_module(module_name, args=None, kwargs=None, type_expr=None)

call_module(module_name, args=None, kwargs=None, type_expr=None) [SOURCE](https://pytorch.ac.cn/docs/stable/_modules/torch/fx/graph.html#Graph.call_module)

在 Graph 中插入一个 call_module Node。一个 call_module 节点表示对 Module 层次结构中 Module 的 forward() 函数的调用。

* 参数
  * **module_name** (字符串)，要调用的 Module 层次结构中 Module 的限定名称。例如，如果跟踪的 Module 具有名为 foo 的子模块，该子模块又具有名为 bar 的子模块，则应将限定名称 foo.bar 作为 module_name 传递以调用该模块。
  * **args** (可选[元组[参数, ...]])，传递给调用方法的位置参数。请注意，这不应包含 self 参数。
  * **kwargs** (可选[字典[字符串, 参数]])，传递给调用方法的关键字参数
  * **type_expr** (可选[Any])，一个可选的类型注解，表示此节点输出的 Python 类型。

* 返回值，新创建并插入的 call_module 节点。

* 返回类型，节点

*此方法与`Graph.create_node()`具有相同的插入点和类型表达式规则。*



#### 0.4 class torch.fx.Node(graph, name, op, target, args, kwargs, return_type=None)

class torch.fx.Node(graph, name, op, target, args, kwargs, return_type=None)[SOURCE](https://pytorch.ac.cn/docs/stable/_modules/torch/fx/node.html#Node)

Node 是表示 Graph 中单个操作的数据结构。在大多数情况下，节点表示对各种实体的调用点，例如运算符、方法和模块（也有一些例外，包括指定函数输入和输出的节点）。每个 Node 都具有由其 op 属性指定的函数。每个 op 值的 Node 语义如下

* `placeholder` 表示函数输入。 `name` 属性指定此值将采用的名称。 `target` 同样是参数的名称。 `args` 包含以下内容之一：
  *  没有任何内容
  *  表示函数输入的默认参数的单个参数
  `kwargs 是无关紧要的。占位符对应于图打印输出中的函数参数（例如 x）。

* `get_attr` 从模块层次结构中检索参数。 `name` 同样是获取结果的名称。 `target` 是参数在模块层次结构中的位置的完全限定名称。 `args` 和 `kwargs` 是无关紧要的。

* `call_function` 将自由函数应用于某些值。 `name` 同样是分配给值的名称。 `target` 是要应用的函数。 `args` 和 `kwargs` 代表函数的参数，遵循 Python 调用约定。

* `call_module` 将模块层次结构中的模块的 `forward()` 方法应用于给定的参数。 `name` 与之前相同。 `target` 是模块层次结构中要调用的模块的完全限定名称。 `args` 和 `kwargs` 代表调用模块的参数，*不包括 self 参数*。

* `call_method` 在值上调用方法。 `name` 与之前相同。 `target` 是要应用于 `self` 参数的方法的字符串名称。 `args` 和 `kwargs` 代表调用模块的参数，*包括 self 参数*。

* `output` 在其 `args[0]` 属性中包含跟踪函数的输出。 这对应于 Graph 打印中的`return`语句。



### 1 编写转换

FX 转换本质上看起来像这样的函数，
```python
import torch
import torch.fx

def transform(m: nn.Module,
              tracer_class : type = torch.fx.Tracer) -> torch.nn.Module:
    # Step 1: Acquire a Graph representing the code in `m`

    # NOTE: torch.fx.symbolic_trace is a wrapper around a call to
    # fx.Tracer.trace and constructing a GraphModule. We'll
    # split that out in our transform to allow the caller to
    # customize tracing behavior.
    graph : torch.fx.Graph = tracer_class().trace(m)

    # Step 2: Modify this Graph or create a new one
    graph = ...

    # Step 3: Construct a Module to return
    return torch.fx.GraphModule(m, graph)
```

FX 转换接收一个 `torch.nn.Module`，从中获取一个 Graph，进行一些修改，并返回一个新的 `torch.nn.Module`。FX 转换返回的 `torch.nn.Module` 与常规 `torch.nn.Module` 相同，可以在另一个 FX 转换中传递它，也可以将其传递给 TorchScript，还可以运行它。确保 FX 转换的输入和输出是 `torch.nn.Module` 将允许可组合性。

也可以修改现有的 GraphModule 而不是创建一个新的，如下所示
```python
import torch
import torch.fx

def transform(m : nn.Module) -> nn.Module:
    gm : torch.fx.GraphModule = torch.fx.symbolic_trace(m)

    # Modify gm.graph
    # <...>

    # Recompile the forward() method of `gm` from its Graph
    gm.recompile()

    return gm
```

这里必须调用 `GraphModule.recompile()` 以使生成的 forward() 方法与修改后的 Graph 同步。

#### 1.1 class torch.fx.GraphModule(*args, **kwargs)

class torch.fx.GraphModule(*args, **kwargs)[SOURCE](https://pytorch.ac.cn/docs/stable/_modules/torch/fx/graph_module.html#GraphModule)

`GraphModule` 是从 `fx.Graph` 生成的 `nn.Module`。Graphmodule 具有 graph 属性，以及从该 graph 生成的 code 和 forward 属性。

当 graph 被重新分配时，code 和 forward 将被自动重新生成。但是，如果在不重新分配 graph 属性本身的情况下编辑 graph 的内容，则必须调用 recompile() 来更新生成的代码。

#### 1.2 recompile()

recompile() [SOURCE](https://pytorch.ac.cn/docs/stable/_modules/torch/fx/graph_module.html#GraphModule.recompile)

* 返回类型: Python 代码

从其 graph 属性重新编译此 GraphModule。这应该在编辑包含的 graph 后调用，否则此 GraphModule 的生成代码将过时。

### 2 关于图 Graph

一个 Graph 是一个数据结构，它表示 GraphModule 上的方法。这需要一些信息：
* 方法的输入
* 方法内部运行的操作
* 方法的输出（即返回值）

这三个信息都用 Node 实例表示。
```python
import torch
import torch.fx

class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.param = torch.nn.Parameter(torch.rand(3, 4))
        self.linear = torch.nn.Linear(4, 5)

    def forward(self, x):
        return torch.topk(torch.sum(
            self.linear(x + self.linear.weight).relu(), dim=-1), 3)

m = MyModule()
gm = torch.fx.symbolic_trace(m)

gm.graph.print_tabular()
```

在这里，定义了一个名为 `MyModule` 的模块并实例化它，对其进行符号跟踪，然后调用 Graph.print_tabular() 方法来打印一个表格，显示此 Graph 的节点。

根据这里打印出来的信息，可以知道，
* 在 FX 中，方法输入通过特殊的 placeholder 节点指定。
* get_attr、call_function、call_module 和 call_method 节点代表方法中的操作
* Graph 中的返回值由一个特殊的 output 节点指定

#### 2.1 print_tabular()

print_tabular() [SOURCE](https://pytorch.org/docs/stable/_modules/torch/fx/graph.html#Graph.print_tabular)

打印表格, 以表格格式打印图的中间表示, 此 API 需要安装 `tabulate` 模块。


### 3 图操作

#### 3.1 直接图操作

构建新的 Graph 的一种方法是直接操作旧的图。为了帮助我们做到这一点，我们可以简单地获取从符号跟踪中获得的 Graph 并对其进行修改。例如，假设我们希望用 torch.mul() 调用替换 torch.add() 调用，
```python
import torch
import torch.fx

# Sample module
class M(torch.nn.Module):
    def forward(self, x, y):
        return torch.add(x, y)

def transform(m: torch.nn.Module,
              tracer_class : type = fx.Tracer) -> torch.nn.Module:
    graph : fx.Graph = tracer_class().trace(m)
    # FX represents its Graph as an ordered list of
    # nodes, so we can iterate through them.
    for node in graph.nodes:
        # Checks if we're calling a function (i.e:
        # torch.add)
        if node.op == 'call_function':
            # The target attribute is the function
            # that call_function calls.
            if node.target == torch.add:
                node.target = torch.mul

    graph.lint() # Does some checks to make sure the
                 # Graph is well-formed.

    return fx.GraphModule(m, graph)
```

还可以进行更复杂的 Graph 重写，例如删除或追加节点。为了帮助进行这些转换，FX 提供了一些用于转换图的实用函数，这些函数在 `class Graph`中可以找到。下面是一个使用这些 API 追加 torch.relu() 调用的示例。
```python
 # Specifies the insertion point. Any nodes added to the
# Graph within this scope will be inserted after `node`
with traced.graph.inserting_after(node):
    # Insert a new `call_function` node calling `torch.relu`
    new_node = traced.graph.call_function(
        torch.relu, args=(node,))

    # We want all places that used the value of `node` to
    # now use that value after the `relu` call we've added.
    # We use the `replace_all_uses_with` API to do this.
    node.replace_all_uses_with(new_node)
```


#### 3.2 使用 replace_pattern() 进行子图重写

FX 还提供了一种基于直接图形操作的自动化级别。 `replace_pattern()` API 本质上是一个用于编辑 Graph 的**查找/替换**工具。它允许指定一个 pattern 和 replacement 函数，它将跟踪这些函数，找到 pattern 图中操作组的实例，并将这些实例替换为 replacement 图的副本。

### 4 代理/重新跟踪

另一种操作 Graph 的方法是重用符号跟踪中使用的 Proxy 机制。

例如，假设我们想要编写一个将 PyTorch 函数分解成更小操作的转换。它将把每个 `F.relu(x)` 调用转换为 `(x > 0) * x`。一种可能性是在 `F.relu` 之后执行必要的图形重写以插入比较和乘法，然后清理原始的 `F.relu`。但是，我们可以使用 Proxy 对象来自动将操作记录到 Graph 中，从而自动化此过程。

要使用此方法，我们将要插入的操作编写为常规 PyTorch 代码，并使用 Proxy 对象作为参数调用该代码。这些 Proxy 对象将捕获对它们执行的操作并将它们追加到 Graph 中。

```python
# Note that this decomposition rule can be read as regular Python
def relu_decomposition(x):
    return (x > 0) * x

decomposition_rules = {}
decomposition_rules[F.relu] = relu_decomposition

def decompose(model: torch.nn.Module,
              tracer_class : type = fx.Tracer) -> torch.nn.Module:
    """
    Decompose `model` into smaller constituent operations.
    Currently,this only supports decomposing ReLU into its
    mathematical definition: (x > 0) * x
    """
    graph : fx.Graph = tracer_class().trace(model)
    new_graph = fx.Graph()
    env = {}
    tracer = torch.fx.proxy.GraphAppendingTracer(new_graph)
    for node in graph.nodes:
        if node.op == 'call_function' and node.target in decomposition_rules:
            # By wrapping the arguments with proxies,
            # we can dispatch to the appropriate
            # decomposition rule and implicitly add it
            # to the Graph by symbolically tracing it.
            proxy_args = [
                fx.Proxy(env[x.name], tracer) if isinstance(x, fx.Node) else x for x in node.args]
            output_proxy = decomposition_rules[node.target](*proxy_args)

            # Operations on `Proxy` always yield new `Proxy`s, and the
            # return value of our decomposition rule is no exception.
            # We need to extract the underlying `Node` from the `Proxy`
            # to use it in subsequent iterations of this transform.
            new_node = output_proxy.node
            env[node.name] = new_node
        else:
            # Default case: we don't have a decomposition rule for this
            # node, so just copy the node over into the new graph.
            new_node = new_graph.node_copy(node, lambda x: env[x.name])
            env[node.name] = new_node
    return fx.GraphModule(model, new_graph)
```

除了避免显式图形操作之外，使用 Proxy 还允许将重写规则指定为原生 Python 代码。对于需要大量重写规则的转换（例如 vmap 或 grad），这通常可以提高规则的可读性和可维护性。

注意，在调用 Proxy 时，我们还传递了一个指向基础变量 graph 的跟踪器。这样做是为了防止图形中的操作是 n 元的（例如，add 是一个二元运算符），对 Proxy 的调用不会创建图形跟踪器的多个实例，因为这会导致意外的运行时错误。建议使用这种使用 Proxy 的方法，尤其是在无法安全地假设基础运算符为一元运算符时。

### 5 解释器模式

在 FX 中，一个有用的代码组织模式是循环遍历 Node 在 Graph 中执行它们。这可以用于多种用途，包括对流经图的值进行运行时分析或通过使用 Proxy 进行代码重跟踪来转换代码。

例如，假设我们想要运行一个 GraphModule 并记录 torch.Tensor 在运行时看到节点上的形状和数据类型属性,
```python
import torch
import torch.fx
from torch.fx.node import Node

from typing import Dict

class ShapeProp:
    """
    Shape propagation. This class takes a `GraphModule`.
    Then, its `propagate` method executes the `GraphModule`
    node-by-node with the given arguments. As each operation
    executes, the ShapeProp class stores away the shape and
    element type for the output values of each operation on
    the `shape` and `dtype` attributes of the operation's
    `Node`.
    """
    def __init__(self, mod):
        self.mod = mod
        self.graph = mod.graph
        self.modules = dict(self.mod.named_modules())

    def propagate(self, *args):
        args_iter = iter(args)
        env : Dict[str, Node] = {}

        def load_arg(a):
            return torch.fx.graph.map_arg(a, lambda n: env[n.name])

        def fetch_attr(target : str):
            target_atoms = target.split('.')
            attr_itr = self.mod
            for i, atom in enumerate(target_atoms):
                if not hasattr(attr_itr, atom):
                    raise RuntimeError(f"Node referenced nonexistant target {'.'.join(target_atoms[:i])}")
                attr_itr = getattr(attr_itr, atom)
            return attr_itr

        for node in self.graph.nodes:
            if node.op == 'placeholder':
                result = next(args_iter)
            elif node.op == 'get_attr':
                result = fetch_attr(node.target)
            elif node.op == 'call_function':
                result = node.target(*load_arg(node.args), **load_arg(node.kwargs))
            elif node.op == 'call_method':
                self_obj, *args = load_arg(node.args)
                kwargs = load_arg(node.kwargs)
                result = getattr(self_obj, node.target)(*args, **kwargs)
            elif node.op == 'call_module':
                result = self.modules[node.target](*load_arg(node.args), **load_arg(node.kwargs))

            # This is the only code specific to shape propagation.
            # you can delete this `if` branch and this becomes
            # a generic GraphModule interpreter.
            if isinstance(result, torch.Tensor):
                node.shape = result.shape
                node.dtype = result.dtype

            env[node.name] = result

        return load_arg(self.graph.result)
```

FX 的完整解释器并不复杂，但它非常有用。为了简化使用这种模式，可以使用 Interpreter 类，它以一种可以覆盖解释器执行的某些方面的方式包含了上述逻辑，方法是通过方法覆盖。

除了执行操作之外，还可以通过将 Proxy 值馈送到解释器来生成一个新的 Graph。Transformer 类包含这种模式。 Transformer 的行为类似于 Interpreter，但不是调用 run 方法从模块获取具体输出值，而是调用 Transformer.transform() 方法返回一个新的 GraphModule。

### 6 调试

在编写转换的过程中，代码可能并不完全正确。在这种情况下，需要进行一些调试。

关键是反向工作：
* 首先，检查调用生成的模块的结果以证明或反驳正确性。
* 然后，检查和调试生成的代码。
* 最后，调试导致生成代码的转换过程。

#### 6.1 转换编写中的常见陷阱

非确定性 set 迭代顺序。

在 Python 中，set 数据类型是无序的。例如，使用 set 来包含像 Node 这样的对象集合会导致意外的非确定性。一个例子是迭代一组 Node 以将它们插入到 Graph 中。因为 set 数据类型是无序的，所以输出程序中操作的顺序将是非确定性的，并且可以在程序调用之间发生变化。

建议的替代方法是使用 dict 数据类型，该类型是插入有序的（Python >= 3.7, cPython >= 3.6）。一个 dict 可以等效地用作一个集合，通过将要进行重复数据删除的值存储在 dict 的键中。

#### 6.2 检查模块的正确性

由于大多数深度学习模块的输出由浮点数 torch.Tensor 实例组成，因此检查两个 torch.nn.Module 结果之间的等效性并不像简单地进行相等性检查那样简单。

```python
import torch
import torch.fx
import torchvision.models as models

def transform(m : torch.nn.Module) -> torch.nn.Module:
    gm = torch.fx.symbolic_trace(m)

    # Imagine we're doing some transforms here
    # <...>

    gm.recompile()

    return gm

resnet18 = models.resnet18()
transformed_resnet18 = transform(resnet18)

input_image = torch.randn(5, 3, 224, 224)

assert resnet18(input_image) == transformed_resnet18(input_image)
"""
RuntimeError: Boolean value of Tensor with more than one value is ambiguous
"""
```

在这里，尝试使用 `==` 等号运算符检查两个深度学习模型的值是否相等。然而，这并不明确，因为该运算符返回的是张量而不是布尔值，而且由于浮点数运算的非交换性，浮点数值的比较应该使用误差范围（或 epsilon）来进行。这里可以使用 torch.allclose() 来代替，它将提供一个近似比较，并考虑相对和绝对容差阈值。

```python
assert torch.allclose(resnet18(input_image), transformed_resnet18(input_image))```
```

#### 6.3 调试生成的代码

由于 FX 在 GraphModule 上生成 forward() 函数，因此使用传统的调试技术（如 print 语句或 pdb）并不像以前那样简单。幸运的是，有几种技术可以用来调试生成的代码。

##### 6.3.1 PDB

调用 pdb 以进入正在运行的程序。虽然表示 Graph 的代码不在任何源文件中，但我们仍然可以使用 pdb 在调用前向传递时手动进入它。

##### 6.3.2 打印生成的代码

如果想要多次运行相同的代码，那么使用 pdb 一步步调试到正确的代码可能会很繁琐。在这种情况下，一种方法是简单地将生成的 forward 代码复制粘贴到代码中，然后从那里进行检查。

```python
# Assume that `traced` is a GraphModule that has undergone some
# number of transforms

# Copy this code for later
print(traced)
# Print the code generated from symbolic tracing. This outputs:
"""
def forward(self, y):
    x = self.x
    add_1 = x + y;  x = y = None
    return add_1
"""

# Subclass the original Module
class SubclassM(M):
    def __init__(self):
        super().__init__()

    # Paste the generated `forward` function (the one we printed and
    # copied above) here
    def forward(self, y):
        x = self.x
        add_1 = x + y;  x = y = None
        return add_1

# Create an instance of the original, untraced Module. Then, create an
# instance of the Module with the copied `forward` function. We can
# now compare the output of both the original and the traced version.
pre_trace = M()
post_trace = SubclassM()
```

##### 6.3.3 使用 GraphModule 中的 to_folder 函数

GraphModule.to_folder() 是 GraphModule 中的一个方法，它允许您将生成的 FX 代码转储到一个文件夹中。虽然将 forward 代码复制到代码中通常就足够了，但使用 to_folder 检查模块和参数可能更容易。
```python
m = symbolic_trace(M())
m.to_folder("foo", "Bar")
from foo import Bar
y = Bar()
```

运行上面的代码后，可以查看 foo/module.py 中的代码，并根据需要进行修改（例如，添加 print 语句或使用 pdb）来调试生成的代码。

#### 6.4 调试转换

现在，我们已经可以确定转换生成的代码是否正确了，是时候调试转换本身了。一旦我们验证了追踪按预期工作，目标就变成了弄清楚我们的 GraphModule 转换过程中出了什么问题。有几种方法可以检查我们追踪的模块。

```python
# Sample Module
class M(torch.nn.Module):
    def forward(self, x, y):
        return x + y

# Create an instance of `M`
m = M()

# Symbolically trace an instance of `M` (returns a GraphModule). In
# this example, we'll only be discussing how to inspect a
# GraphModule, so we aren't showing any sample transforms for the
# sake of brevity.
traced = symbolic_trace(m)

# Print the code produced by tracing the module.
print(traced)
# The generated `forward` function is:
"""
def forward(self, x, y):
    add = x + y;  x = y = None
    return add
"""

# Print the internal Graph.
print(traced.graph)
# This print-out returns:
"""
graph():
    %x : [num_users=1] = placeholder[target=x]
    %y : [num_users=1] = placeholder[target=y]
    %add : [num_users=1] = call_function[target=operator.add](args = (%x, %y), kwargs = {})
    return add
"""

# Print a tabular representation of the internal Graph.
traced.graph.print_tabular()
# This gives us:
"""
opcode         name    target                   args    kwargs
-------------  ------  -----------------------  ------  --------
placeholder    x       x                        ()      {}
placeholder    y       y                        ()      {}
call_function  add     <built-in function add>  (x, y)  {}
output         output  output                   (add,)  {}
"""
```

使用上面的实用函数，我们可以比较我们在应用转换之前和之后追踪的模块。有时，简单的视觉比较足以追踪到错误。如果仍然不清楚出了什么问题，就需要借助 PDB 来调试查找问题了。

参考上面的例子，考虑以下代码，
```python
# Sample user-defined function
def transform_graph(module: torch.nn.Module, tracer_class : type = fx.Tracer) -> torch.nn.Module:
    # Get the Graph from our traced Module
    g = tracer_class().trace(module)

    """
    Transformations on `g` go here
    """

    return fx.GraphModule(module, g)

# Transform the Graph
transformed = transform_graph(traced)

# Print the new code after our transforms. Check to see if it was
# what we expected
print(transformed)
```

使用上面的例子，假设对`print(traced)` 的调用显示我们的转换中存在错误。我们可以使用调试器找出问题所在。启动一个 pdb 会话，可以通过在 `transform_graph(traced)` 上断点，然后单步执行对 `transform_graph(traced)` 的调用来查看转换期间发生了什么。也可以通过编辑`print_tabular`方法来打印图中节点的不同属性。

### 7 符号跟踪的局限性

FX 使用一个 **符号跟踪** 系统（也称为 符号执行）来捕获程序语义的可转换/可分析形式。该系统是 **跟踪** 的，因为它执行程序（实际上是一个 torch.nn.Module 或函数）来记录操作。同时它也是 **符号** 的，因为在执行过程中流经程序的数据不是真实数据，而是符号（在 FX 中称为 Proxy）。虽然符号跟踪适用于大多数神经网络代码，但它也有一些局限性。

#### 7.1 动态控制流

符号追踪的主要限制是它目前不支持**动态控制流**。也就是说，循环或if语句，其条件可能取决于程序的输入值。

```python
def func_to_trace(x):
    if x.sum() > 0:
        return torch.relu(x)
    else:
        return torch.neg(x)

traced = torch.fx.symbolic_trace(func_to_trace)
"""
  <...>
  File "dyn.py", line 6, in func_to_trace
    if x.sum() > 0:
  File "pytorch/torch/fx/proxy.py", line 155, in __bool__
    return self.tracer.to_bool(self)
  File "pytorch/torch/fx/proxy.py", line 85, in to_bool
    raise TraceError('symbolically traced variables cannot be used as inputs to control flow')
torch.fx.proxy.TraceError: symbolically traced variables cannot be used as inputs to control flow
"""
```
#### 7.2 静态控制流

所谓的**静态控制流**是支持的。静态控制流是循环或if语句，其值在调用之间不会改变。通常，在 PyTorch 程序中，这种控制流是针对根据超参数做出模型架构决策的代码而产生的。
```python
import torch
import torch.fx

class MyModule(torch.nn.Module):
    def __init__(self, do_activation : bool = False):
        super().__init__()
        self.do_activation = do_activation
        self.linear = torch.nn.Linear(512, 512)

    def forward(self, x):
        x = self.linear(x)
        # This if-statement is so-called static control flow.
        # Its condition does not depend on any input values
        if self.do_activation:
            x = torch.relu(x)
        return x

without_activation = MyModule(do_activation=False)
with_activation = MyModule(do_activation=True)

traced_without_activation = torch.fx.symbolic_trace(without_activation)
print(traced_without_activation.code)
"""
def forward(self, x):
    linear_1 = self.linear(x);  x = None
    return linear_1
"""

traced_with_activation = torch.fx.symbolic_trace(with_activation)
print(traced_with_activation.code)
"""
import torch
def forward(self, x):
    linear_1 = self.linear(x);  x = None
    relu_1 = torch.relu(linear_1);  linear_1 = None
    return relu_1
"""
```

if 语句`if self.do_activation`不依赖于任何函数输入，因此它是静态的。`do_activation`可以被认为是一个超参数，并且具有不同参数值的`MyModule`的不同实例的跟踪具有不同的代码。这是一个有效的模式，符号追踪支持。

许多动态控制流的实例在语义上是静态控制流。这些实例可以通过消除对输入值的依赖关系来支持符号追踪，例如，将值移动到`Module`属性，或在符号追踪期间将具体值绑定到参数。

```python
def f(x, flag):
    if flag: return x
    else: return x*2

fx.symbolic_trace(f) # Fails!

fx.symbolic_trace(f, concrete_args={'flag': True})
```

在真正动态控制流的情况下，包含此代码的程序部分可以被追踪为对方法的调用或函数，而不是通过它们进行追踪。


### 8 其他

#### 8.1 非-torch 函数

FX 使用`__torch_function__`作为其拦截调用的机制。某些函数，例如内置 Python 函数或 math 模块中的函数，不受 `__torch_function__`涵盖，但我们仍然希望在符号跟踪中捕获它们。例如
```python
import torch
import torch.fx
from math import sqrt

def normalize(x):
    """
    Normalize `x` by the size of the batch dimension
    """
    return x / sqrt(len(x))

# It's valid Python code
normalize(torch.rand(3, 4))

traced = torch.fx.symbolic_trace(normalize)
"""
  <...>
  File "sqrt.py", line 9, in normalize
    return x / sqrt(len(x))
  File "pytorch/torch/fx/proxy.py", line 161, in __len__
    raise RuntimeError("'len' is not supported in symbolic tracing by default. If you want "
RuntimeError: 'len' is not supported in symbolic tracing by default. If you want this call to be recorded, please call torch.fx.wrap('len') at module scope
"""
```
错误信息很清楚，内置函数 len 不受支持。我们可以使用 wrap() API 使此类函数在跟踪中作为直接调用被记录下来。
```python
torch.fx.wrap('len')
torch.fx.wrap('sqrt')

traced = torch.fx.symbolic_trace(normalize)

print(traced.code)
"""
import math
def forward(self, x):
    len_1 = len(x)
    sqrt_1 = math.sqrt(len_1);  len_1 = None
    truediv = x / sqrt_1;  x = sqrt_1 = None
    return truediv
"""
```

#### 8.2 使用 Tracer 类自定义跟踪

Tracer 类是 symbolic_trace 实现的基础类。可以通过子类化 Tracer 来自定义跟踪的行为，如下所示
```python
class MyCustomTracer(torch.fx.Tracer):
    # Inside here you can override various methods
    # to customize tracing. See the `Tracer` API
    # reference
    pass


# Let's use this custom tracer to trace through this module
class MyModule(torch.nn.Module):
    def forward(self, x):
        return torch.relu(x) + torch.ones(3, 4)

mod = MyModule()

traced_graph = MyCustomTracer().trace(mod)
# trace() returns a Graph. Let's wrap it up in a
# GraphModule to make it runnable
traced = torch.fx.GraphModule(mod, traced_graph)
```

##### 8.2.1 叶模块

叶模块是在符号跟踪中显示为调用而不是被跟踪的模块。默认的叶模块集是标准 `torch.nn` 模块实例集。例如
```python
class MySpecialSubmodule(torch.nn.Module):
    def forward(self, x):
        return torch.neg(x)

class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 4)
        self.submod = MySpecialSubmodule()

    def forward(self, x):
        return self.submod(self.linear(x))

traced = torch.fx.symbolic_trace(MyModule())
print(traced.code)
# `linear` is preserved as a call, yet `submod` is traced though.
# This is because the default set of "Leaf Modules" includes all
# standard `torch.nn` modules.
"""
import torch
def forward(self, x):
    linear_1 = self.linear(x);  x = None
    neg_1 = torch.neg(linear_1);  linear_1 = None
    return neg_1
"""
```

可以通过覆盖 `Tracer.is_leaf_module()` 来自定义叶模块集。

#### 8.3 注意事项

* 张量构造函数（例如 `torch.zeros、torch.ones、torch.rand、torch.randn、torch.sparse_coo_tensor`）目前不可跟踪。
  * 确定性构造函数（zeros，ones）可以使用，它们产生的值将作为常量嵌入到跟踪中。只有当这些构造函数的参数引用动态输入大小时，这才会出现问题。在这种情况下，`ones_like` 或 `zeros_like` 可能是一个可行的替代方案。
  * 非确定性构造函数（rand，randn）将在跟踪中嵌入单个随机值。这可能不是预期的行为。一种解决方法是将 `torch.randn` 包裹在一个 `torch.fx.wrap` 函数中，并调用该函数。
    ```python
    @torch.fx.wrap
    def torch_randn(x, shape):
        return torch.randn(shape)

    def f(x):
        return x + torch_randn(x, 5)
    fx.symbolic_trace(f)
    ```

* 类型注释
  * 支持 Python 3 风格的类型注释（例如 `func(x : torch.Tensor, y : int) -> torch.Tensor`），并且符号跟踪将保留它们
  * 目前不支持 Python 2 风格的注释类型注释 `# type: (torch.Tensor, int) -> torch.Tensor`。
  * 目前不支持函数内局部名称的注释。

* 关于 training 标志和子模块的注意事项
  * 当使用诸如 `torch.nn.functional.dropout` 之类的函数时，训练参数通常作为 `self.training` 传入。在 FX 跟踪期间，这很可能被烘焙为一个常量值。
    ```python
    import torch
    import torch.fx

    class DropoutRepro(torch.nn.Module):
      def forward(self, x):
        return torch.nn.functional.dropout(x, training=self.training)


    traced = torch.fx.symbolic_trace(DropoutRepro())
    print(traced.code)
    """
    def forward(self, x):
      dropout = torch.nn.functional.dropout(x, p = 0.5, training = True, inplace = False);  x = None
      return dropout
    """

    traced.eval()

    x = torch.randn(5, 3)
    torch.testing.assert_close(traced(x), x)
    """
    AssertionError: Tensor-likes are not close!

    Mismatched elements: 15 / 15 (100.0%)
    Greatest absolute difference: 1.6207983493804932 at index (0, 2) (up to 1e-05 allowed)
    Greatest relative difference: 1.0 at index (0, 0) (up to 0.0001 allowed)
    """
    ```

  * 但是，当使用标准 `nn.Dropout()` 子模块时，训练标志被封装，并且由于 `nn.Module` 对象模型的保留，可以更改。
    ```python
    class DropoutRepro2(torch.nn.Module):
      def __init__(self):
        super().__init__()
        self.drop = torch.nn.Dropout()

      def forward(self, x):
        return self.drop(x)

    traced = torch.fx.symbolic_trace(DropoutRepro2())
    print(traced.code)
    """
    def forward(self, x):
      drop = self.drop(x);  x = None
      return drop
    """

    traced.eval()

    x = torch.randn(5, 3)
    torch.testing.assert_close(traced(x), x)
    ```

  * 由于这种差异，可以考虑将与 training 标志动态交互的模块标记为叶模块。
