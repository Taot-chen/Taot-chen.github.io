---
layout: post
title: multi-card_inference_LLM_with_tensor-parallel
date: 2024-03-02
tags: [LLM]
author: taot
---

## 用 tensor-parallel 多卡并发推理大模型


利用 tensor-parallel 把模型训练与推理的 workload 平均分布到多块 GPU，一方面可以提高推理速度，另一方面 vram 的负载平衡也让复杂的 prompt 能被轻松处理。

import 相关的 libs：
```python
# torch version 2.0.0
import torch
# tensor-parallel version 1.0.22
from tensor_parallel import TensorParallelPreTrainedModel
# transformer version 4.28.0.dev0
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig
```

加载 LLaMA-7B 并转化为 TensorParallelPreTrainedModel：
```python
model = LlamaForCausalLM.from_pretrained("./llama-7b-hf", torch_dtype=torch.float16)
model = TensorParallelPreTrainedModel(model, ["cuda:0", "cuda:1"])
```

加载 tokenizer 并进行推理：
```python
tokenizer = LlamaTokenizer.from_pretrained("./llama-7b-hf")

tokens = tokenizer("Hi, how are you?", return_tensors="pt")
tokenizer.decode(model.generate(tokens["input_ids"].cuda(0), attention_mask=tokens["attention_mask"].cuda(0))[0])

# 输出：
#  'Hi, how are you? I'm a 20 year old girl from the Netherlands'

tokens = tokenizer("Once upon a time, there was a lonely computer ", return_tensors="pt")
tokenizer.decode(model.generate(tokens["input_ids"].cuda(0), attention_mask=tokens["attention_mask"].cuda(0), max_length=256)[0])

# 输出：
#  'Once upon a time, there was a lonely computer. It was a very old computer, and it had been sitting in a box for a long time. It was very sad, because it had no friends.\nOne day, a little girl came to the computer. She was very nice, and she said, “Hello, computer. I’m going to be your friend.”\nThe computer was very happy. It said, “Thank you, little girl. I’m very happy to have you as my friend.”\nThe little girl said, “I’m going to call you ‘Computer.’”\n“That’s a good name,” said Computer.\nThe little girl said, “I’m going to teach you how to play games.”\n“That’s a good idea,” said Computer.\nThe little girl said, “I’m going to teach you how to do math.”\nThe little girl said, “I’m going to teach you how to write stories.”\nThe little girl said, “I’m going to teach you how to draw pictures.”\nThe little girl said, “I’m going to teach you how to play music.”\nThe little girl said, “I’m'
```

在这里，我们把我们的推理逻辑平均分布到了两块 GPU 上。


> tensor parallel 在主流的推理框架已经很好的支持了，[vLLM](https://github.com/vllm-project/vllm) 和 [lightllm](https://github.com/ModelTC/lightllm) 都是很好的选择。

