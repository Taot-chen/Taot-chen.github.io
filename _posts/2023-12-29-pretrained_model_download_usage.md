---
layout: post
title: pretrained_model_download_usage
date: 2023-12-28
---

## 预训练模型下载和使用

### 1 huggingface
Huggingface是一家公司，在Google发布BERT模型不久之后，这家公司推出了BERT的pytorch实现，形成一个开源库pytorch-pretrained-bert。后续又实现了其他的预训练模型，如GPT、GPT2、ToBERTa、T5等，便把开源库的名字改成transformers，transformers包括各种模型的 pytorch 实现

Google发布的原始BERT预训练模型（训练好的参数）是基于Tensorflow的，Huggingface是基于pytorch的。

### 2 预训练模型下载
huggingface官网的这个链接：[huggingface 官网](https://huggingface.co/models) 可以搜索到当前支持的模型，但是目前正常情况下应该是访问不到了。可以使用 [魔塔社区](https://www.modelscope.cn/home) 来获取模型并下载，获得离线模型之后再本地调试

modelscope 的模型下载可以使用**模型文件**页面提供的 SDK 下载脚本来下载模型文件和 checkpoint 到本地，例如：
```python
    #模型下载
    from modelscope import snapshot_download
    model_dir = snapshot_download('ZhipuAI/GLM130B')
```
网络结构主要分为两种：Base 和 Large。Base版本相比于Large版本网络规模较小。

Uncased和Cased：Uncased模型在WordPiece tokenization之前会转换成小写,如John Smith变为john smith. Uncased模型还会去除发音标记（accent markers）.Cased模型保留了大小写和发音标记。一般来说Uncased模型更好。但是在case信息也有作用的场景下，就应该使用cased模型 (e.g., Named Entity Recognition or Part-of-Speech tagging)

Multilingual表示预训练时使用多种语言的语料，Chinese表示预训练时使用中文，没有额外标注的则表示用英文语料训练。

Whole Word Masking表示整词掩盖而不是只掩盖词语的一部分。

ext差别是增加了训练数据集同时也增加了训练步数。

### 3 模型文件说明
关键的几个文件：
* 配置文件，config.json
* 词典文件，vocab.json或vocab.txt
* 预训练模型文件（checkpoint），pytorch_model.bin或tf_model.h5
* 另外，对于 transformers 还没有支持的大模型，还会包含该模型自定义的 modeling_xxx.py 脚本，用于构建模型

### 4 预训练模型的使用
transformers 库主要的几个类：
* model
* tokenizer
* configuration
* 其他类如 GPT2LMHeadModel 都是对应类的子类

这三个类，所有相关的类都衍生自这三个类，他们都有from_pretained()方法和save_pretrained()方法

from_pretrained方法的第一个参数都是pretrained_model_name_or_path，这个参数设置为我们下载的文件目录即可

使用的基本原理也非常简单，from_pretrained的参数pretrained_model_name_or_path，可以接受的参数有几种，short-cut name（缩写名称，类似于gpt2这种）、identifier name（类似于microsoft/DialoGPT-small这种）、文件夹、文件

对于short-cut name或identifier name，这种情况下，本地有文件，可以使用本地的，本地没有文件，则下载。一些常用的short-cut name，可以从这个链接查看：https://huggingface.co/transformers/pretrained_models.html

对于文件夹，则会从文件夹中找vocab.json、pytorch_model.bin、tf_model.h5、merges.txt、special_tokens_map.json、added_tokens.json、tokenizer_config.json、sentencepiece.bpe.model等进行加载。这也是为什么下载的时候，一定要保证这些名称是这几个，不能变。

对于文件，则会直接加载文件。

#### 1）使用方式
* 指定模型名字
  这种方式不需要下载预训练模型，函数调用过程中如果发现没有这个模型就会自动下载
  ```python
    from transformers import AutoTokenizer, AutoModel
    tokenizer = AutoTokenizer.from_pretrained("nghuyong/ernie-1.0")
    model = AutoModel.from_pretrained("nghuyong/ernie-1.0")
  ```
* 指定模型路径
  这种方式需要先下载好预训练模型的文件
  ```python
    from transformers import AutoTokenizer, AutoModel
    tokenizer = AutoTokenizer.from_pretrained("/home/models/huggingface/gpt2")
    model = AutoModel.from_pretrained("/home/models/huggingface/gpt2")
  ```

#### 2）调用
传入参数input_ids和attention_mask，返回最后一层的输出last_hidden_state，形状为(batch_size,sequence_len,hidden_size)以及[CLS]对应的输出pooled_output，形状为(batch_size,hidden_size)
```python
    last_hidden_state, pooled_output = bert_model(
        input_ids=encoding['input_ids'],
        attention_mask=encoding['attention_mask']
    )
```
