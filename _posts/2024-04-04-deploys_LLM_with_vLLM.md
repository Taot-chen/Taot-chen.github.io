---
layout: post
title: deploys_LLM_with_vLLM
date: 2024-04-04
tags: [llm]
author: taot
---


## vLLM 部署大模型



### 1 介绍


vLLM 是来自 UC Berkeley 的 LMSYS 在 LLM 推理方面的最新工作（没错就是搞出 Vicuna 的那个 group），最大亮点是采用 Paged Attention 技术，结合 Continuous Batching，极大地优化了 realtime 场景下的 LLM serving 的 throughput 与内存使用。

[vllm github 仓库](https://github.com/vllm-project/vllm/tree/main)

#### 1.1 安装

安装命令：
```bash
pip3 install vllm

# vllm==0.2.7
# transformers==4.36.2
# requests==2.31.0
# gradio==4.14.0
```

### 2 使用

#### 2.1 线下批量推理

线下批量推理：为输入的prompts列表，使用vLLM生成答案。

```bash
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"

from vllm import LLM, SamplingParams

llm = LLM('/data-ai/model/llama2/llama2_hf/Llama-2-13b-chat-hf')

INFO 01-18 08:13:26 llm_engine.py:70] Initializing an LLM engine with config: model='/data-ai/model/llama2/llama2_hf/Llama-2-13b-chat-hf', tokenizer='/data-ai/model/llama2/llama2_hf/Llama-2-13b-chat-hf', tokenizer_mode=auto, revision=None, tokenizer_revision=None, trust_remote_code=False, dtype=torch.float16, max_seq_len=4096, download_dir=None, load_format=auto, tensor_parallel_size=1, quantization=None, enforce_eager=False, seed=0)
INFO 01-18 08:13:37 llm_engine.py:275] # GPU blocks: 3418, # CPU blocks: 327
INFO 01-18 08:13:39 model_runner.py:501] Capturing the model for CUDA graphs. This may lead to unexpected consequences if the model is not static. To run the model in eager mode, set 'enforce_eager=True' or use '--enforce-eager' in the CLI.
INFO 01-18 08:13:39 model_runner.py:505] CUDA graphs can take additional 1~3 GiB memory per GPU. If you are running out of memory, consider decreasing `gpu_memory_utilization` or enforcing eager mode.
INFO 01-18 08:13:44 model_runner.py:547] Graph capturing finished in 5 secs.

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

outputs = llm.generate(prompts, sampling_params)

# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

Processed prompts: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4/4 [00:00<00:00, 11.76it/s]

Prompt: 'Hello, my name is', Generated text: " Sherry and I'm a stay at home mom of three beautiful children."
Prompt: 'The president of the United States is', Generated text: ' one of the most powerful people in the world, and yet, many people do'
Prompt: 'The capital of France is', Generated text: ' Paris. This is a fact that is well known to most people, but there'
Prompt: 'The future of AI is', Generated text: ' likely to be shaped by a combination of technological advancements and soci'
```

#### 2.2 API Server服务

vLLM可以部署为API服务，web框架使用FastAPI。API服务使用AsyncLLMEngine类来支持异步调用。

使用命令 python -m vllm.entrypoints.api_server --help 可查看支持的脚本参数。

API服务启动命令:

```bash
CUDA_VISIBLE_DEVICES=6,7 python -m vllm.entrypoints.api_server --model /data-ai/model/llama2/llama2_hf/Llama-2-13b-chat-hf
```

输入：

```bash
curl http://localhost:8000/generate \
    -d '{
        "prompt": "San Francisco is a",
        "use_beam_search": true,
        "n": 4,
        "temperature": 0
    }'
```

输出：

```bash
{
    "text": [
        "San Francisco is a city of neighborhoods, each with its own unique character and charm. Here are",
        "San Francisco is a city in California that is known for its iconic landmarks, vibrant",
        "San Francisco is a city of neighborhoods, each with its own unique character and charm. From the",
        "San Francisco is a city in California that is known for its vibrant culture, diverse neighborhoods"
    ]
}
```

#### 2.3 OpenAI风格的API服务

启动命令：

```bash
CUDA_VISIBLE_DEVICES=6,7 python -m vllm.entrypoints.openai.api_server --model /data-ai/model/llama2/llama2_hf/Llama-2-13b-chat-hf --served-model-name llama-2-13b-chat-hf
```

并且还可指定对话模板（chat-template）。

##### 2.3.1 查看模型

```bash
curl http://localhost:8000/v1/models
```

输出：

```bash
{
   "object": "list",
   "data": [
     {
       "id": "llama-2-13b-chat-hf",
       "object": "model",
       "created": 1705568412,
       "owned_by": "vllm",
       "root": "llama-2-13b-chat-hf",
      "parent": null,
      "permission": [
        {
          "id": "modelperm-d7ca4aa0eee44eb4a50e37eba06e520d",
          "object": "model_permission",
          "created": 1705568412,
          "allow_create_engine": false,
          "allow_sampling": true,
          "allow_logprobs": true,
          "allow_search_indices": false,
          "allow_view": true,
          "allow_fine_tuning": false,
          "organization": "*",
          "group": null,
          "is_blocking": false
        }
      ]
    }
  ]
}
```

##### 2.3.2 text completion

输入：
```bash
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "llama-2-13b-chat-hf",
        "prompt": "San Francisco is a",
        "max_tokens": 7,
        "temperature": 0
    }' | jq .
```

输出：
```bash
{
   "id": "cmpl-d1ba6b9f1551443e87d80258a3bedad1",
   "object": "text_completion",
   "created": 19687093,
   "model": "llama-2-13b-chat-hf",
   "choices": [
     {
       "index": 0,
       "text": " city that is known for its v",
      "logprobs": null,
      "finish_reason": "length"
    }
  ],
  "usage": {
    "prompt_tokens": 5,
    "total_tokens": 12,
    "completion_tokens": 7
  }
}
```

##### 2.3.3 chat completion

输入：
```bash
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "llama-2-13b-chat-hf",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Who won the world series in 2020?"}
        ]
    }' | jq .
```

输出：
```bash
{
   "id": "cmpl-94fc8bc170be4c29982a08aa6f01e298",
   "object": "chat.completion",
   "created": 19687353,
   "model": "llama-2-13b-chat-hf",
   "choices": [
     {
       "index": 0,
       "message": {
        "role": "assistant",
        "content": "  Hello! I'm happy to help! The Washington Nationals won the World Series in 2020. They defeated the Houston Astros in Game 7 of the series, which was played on October 30, 2020."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 40,
    "total_tokens": 95,
    "completion_tokens": 55
  }
}
```

### 3 vLLM 实践

#### 3.1 离线推理

```python
from vllm import LLM

prompts = ["Hello, my name is", "The capital of France is"]  # Sample prompts.
llm = LLM(model="lmsys/vicuna-7b-v1.3")  # Create an LLM.
outputs = llm.generate(prompts)  # Generate texts from the prompts.
```

#### 3.2 在线服务启动

```bash
python -m vllm.entrypoints.openai.api_server --model lmsys/vicuna-7b-v1.3
```

##### 3.2.1 在线服务调用

```bash
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "lmsys/vicuna-7b-v1.3",
        "prompt": "San Francisco is a",
        "max_tokens": 7,
        "temperature": 0
    }'
```

----------
#### 3.3 大模型简单问答

vLLM暂不支持同时部署多个大模型，但是可以采用一次部署一个模型，部署多次的方法来实现部署多个大模型，这里采用llama-2-13b-chat-hf和Baichuan2-13B-Chat.

模型部署的命令如下：

```bash
CUDA_VISIBLE_DEVICES=6 python -m vllm.entrypoints.openai.api_server --host 0.0.0.0 --port 50072 --model /data-ai/model/llama2/llama2_hf/Llama-2-13b-chat-hf --served-model-name llama-2-13b-chat-hf

CUDA_VISIBLE_DEVICES=7 python -m vllm.entrypoints.openai.api_server --host 0.0.0.0 --port 50073 --model /data-ai/model/baichuan2/Baichuan2-13B-Chat --served-model-name Baichuan2-13B-Chat --trust-remote-code --chat-template /data-ai/usr/code/template_baichuan.jinja
```

其中，template_baichuan.jinja（对话模板）采用vLLM在github官方网站中的examples文件夹下的同名文件。

使用Gradio来构建页面，主要实现大模型问答功能，Python代码如下：

```python
 # -*- coding: utf-8 -*-
 # @place: Pudong, Shanghai
 # @file: gradio_for_llm.py
 # @time: 2024/1/19 13:30
 import gradio as gr
 import requests
 
 models = ['llama-2-13b-chat-hf', 'Baichuan2-13B-Chat']
 

def completion(question):
    model_url_dict = {models[0]: "http://localhost:50072/v1/chat/completions",
                      models[1]: "http://localhost:50073/v1/chat/completions",
                      }
    answers = []
    for model in models:
        headers = {'Content-Type': 'application/json'}

        json_data = {
            'model': model,
            'messages': [
                {
                    'role': 'system',
                    'content': 'You are a helpful assistant.'
                },
                {
                    'role': 'user',
                    'content': question
                },
            ],
        }

        response = requests.post(model_url_dict[model], headers=headers, json=json_data)
        answer = response.json()["choices"][0]["message"]["content"]
        answers.append(answer)
    return answers


demo = gr.Interface(
    fn=completion,
    inputs=gr.Textbox(lines=5, placeholder="input your question", label="question"),
    outputs=[gr.Textbox(lines=5, placeholder="answer", label=models[0]),
             gr.Textbox(lines=5, placeholder="answer", label=models[1])]
)

demo.launch(server_name='0.0.0.0', share=True)
```

#### 3.4 大模型输出TPS

衡量大模型部署工具的指标之一为TPS（Token Per Second），即每秒模型输出的token数量。

我们以llama-2-13b-chat-hf，测试数据集参考网站中的问题集:[https://modal.com/docs/examples/vllm_inference](https://modal.com/docs/examples/vllm_inference)，，一共59个问题。

Python代码如下：

```python
# -*- coding: utf-8 -*-
  # @place: Pudong, Shanghai
  # @file: gradio_for_throughput.py
  # @time: 2024/1/19 16:05
  import gradio as gr
  import requests
  import time
  
  questions = [
         # Coding questions
         "Implement a Python function to compute the Fibonacci numbers.",
         "Write a Rust function that performs binary exponentiation.",
         "How do I allocate memory in C?",
         "What are the differences between Javascript and Python?",
         "How do I find invalid indices in Postgres?",
         "How can you implement a LRU (Least Recently Used) cache in Python?",
         "What approach would you use to detect and prevent race conditions in a multithreaded application?",
         "Can you explain how a decision tree algorithm works in machine learning?",
         "How would you design a simple key-value store database from scratch?",
         "How do you handle deadlock situations in concurrent programming?",
         "What is the logic behind the A* search algorithm, and where is it used?",
         "How can you design an efficient autocomplete system?",
         "What approach would you take to design a secure session management system in a web application?",
         "How would you handle collision in a hash table?",
         "How can you implement a load balancer for a distributed system?",
         # Literature
         "What is the fable involving a fox and grapes?",
         "Write a story in the style of James Joyce about a trip to the Australian outback in 2083, to see robots in the beautiful desert.",
         "Who does Harry turn into a balloon?",
         "Write a tale about a time-traveling historian who's determined to witness the most significant events in human history.",
         "Describe a day in the life of a secret agent who's also a full-time parent.",
         "Create a story about a detective who can communicate with animals.",
         "What is the most unusual thing about living in a city floating in the clouds?",
         "In a world where dreams are shared, what happens when a nightmare invades a peaceful dream?",
         "Describe the adventure of a lifetime for a group of friends who found a map leading to a parallel universe.",
         "Tell a story about a musician who discovers that their music has magical powers.",
         "In a world where people age backwards, describe the life of a 5-year-old man.",
         "Create a tale about a painter whose artwork comes to life every night.",
         "What happens when a poet's verses start to predict future events?",
         "Imagine a world where books can talk. How does a librarian handle them?",
         "Tell a story about an astronaut who discovered a planet populated by plants.",
         "Describe the journey of a letter traveling through the most sophisticated postal service ever.",
         "Write a tale about a chef whose food can evoke memories from the eater's past.",
         # History
         "What were the major contributing factors to the fall of the Roman Empire?",
         "How did the invention of the printing press revolutionize European society?",
         "What are the effects of quantitative easing?",
         "How did the Greek philosophers influence economic thought in the ancient world?",
         "What were the economic and philosophical factors that led to the fall of the Soviet Union?",
         "How did decolonization in the 20th century change the geopolitical map?",
         "What was the influence of the Khmer Empire on Southeast Asia's history and culture?",
         # Thoughtfulness
         "Describe the city of the future, considering advances in technology, environmental changes, and societal shifts.",
         "In a dystopian future where water is the most valuable commodity, how would society function?",
         "If a scientist discovers immortality, how could this impact society, economy, and the environment?",
         "What could be the potential implications of contact with an advanced alien civilization?",
         # Math
         "What is the product of 9 and 8?",
         "If a train travels 120 kilometers in 2 hours, what is its average speed?",
         "Think through this step by step. If the sequence a_n is defined by a_1 = 3, a_2 = 5, and a_n = a_(n-1) + a_(n-2) for n > 2, find a_6.",
         "Think through this step by step. Calculate the sum of an arithmetic series with first term 3, last term 35, and total terms 11.",
         "Think through this step by step. What is the area of a triangle with vertices at the points (1,2), (3,-4), and (-2,5)?",
         "Think through this step by step. Solve the following system of linear equations: 3x + 2y = 14, 5x - y = 15.",
         # Facts
         "Who was Emperor Norton I, and what was his significance in San Francisco's history?",
         "What is the Voynich manuscript, and why has it perplexed scholars for centuries?",
         "What was Project A119 and what were its objectives?",
         "What is the 'Dyatlov Pass incident' and why does it remain a mystery?",
         "What is the 'Emu War' that took place in Australia in the 1930s?",
         "What is the 'Phantom Time Hypothesis' proposed by Heribert Illig?",
         "Who was the 'Green Children of Woolpit' as per 12th-century English legend?",
         "What are 'zombie stars' in the context of astronomy?",
         "Who were the 'Dog-Headed Saint' and the 'Lion-Faced Saint' in medieval Christian traditions?",
         "What is the story of the 'Globsters', unidentified organic masses washed up on the shores?",
     ]
 
 
 def chat_completion(question):
     url = "http://localhost:50072/v1/chat/completions"
 
     headers = {'Content-Type': 'application/json'}
 
     json_data = {
         'model': "llama-2-13b-chat-hf",
         'messages': [
             {
                 'role': 'system',
                 'content': 'You are a helpful assistant.'
             },
             {
                 'role': 'user',
                 'content': question
             },
         ],
     }
 
     response = requests.post(url, headers=headers, json=json_data)
     answer = response.json()["choices"][0]["message"]["content"]
     output_tokens = response.json()["usage"]["completion_tokens"]
    return answer, output_tokens


def slowly_reverse(texts, progress=gr.Progress()):
    total_token_cnt = 0
    progress(0, desc="starting...")
    q_list = texts.split('\n')
    s_time = time.time()
    data_list = []
    for q in progress.tqdm(q_list, desc=f"generating..."):
        answer, output_token = chat_completion(q)
        total_token_cnt += output_token
        data_list.append([q, answer[:50], total_token_cnt/(time.time() - s_time)])
        print(f"{total_token_cnt/(time.time() - s_time)} TPS")

    return data_list


demo = gr.Interface(
    fn=slowly_reverse,
    # 自定义输入框
    inputs=gr.Textbox(value='\n'.join(questions), label="questions"),
    # 设置输出组件
    outputs=gr.DataFrame(label='Table', headers=['question', 'answer', 'TPS'], interactive=True, wrap=True)
)

demo.queue().launch(server_name='0.0.0.0', share=True)
```


### 4 vLLM 离线推理流程
### 4.1 Create sampling_params

根据用户设置，创建采样参数类对象 sampling_params，只指定 temperature=0.8, top_p=0.95 的情况下，其他默认值如下所示：

```python
SamplingParams(n=1,
               best_of=1,
               presence_penalty=0.0,
               frequency_penalty=0.0,
               temperature=0.8,
               top_p=0.95,
               top_k=-1,
               use_beam_search=False,
               stop=[],
               ignore_eos=False,
               max_tokens=16,
               logprobs=None)
```

### 4.2 Create an LLM

LLM 类对象的构造函数中，首先创建 EngineArgs 类对象 engine_args 如下：

```python
EngineArgs(model='/bigdata/shared/models/huggingface/opt-125m',
           tokenizer='/bigdata/shared/models/huggingface/opt-125m',
           tokenizer_mode='auto',
           trust_remote_code=False,
           download_dir=None,
           use_np_weights=False,
           use_dummy_weights=False,
           dtype='auto',
           seed=0,
           worker_use_ray=False,
           pipeline_parallel_size=1,
           tensor_parallel_size=1,
           block_size=16,
           swap_space=4,
           gpu_memory_utilization=0.9,
           max_num_batched_tokens=2560,
           max_num_seqs=256,
           disable_log_stats=True,
           quant_mode=None)
```

然后基于 engine_args ，构造 LLM 类内核心变量 llm_engine ，最后添加一个类内计数器 request_counter
```python
self.llm_engine = LLMEngine.from_engine_args(engine_args)
self.request_counter = Counter()
```

### 4.3 Generate

在 LLM.generate 的处理过程中，核心操作分为两步。

第一步是调用 LLM._add_request ，通过 LLM.llm_engine.add_request 将用户传入的请求添加到请求列表中，添加完后，请求列表 LLM.llm_engine.scheduler.waiting 中内容如下：

```python
[ \
    SequenceGroup(request_id=0, sampling_params=SamplingParams(n=1, best_of=1, presence_penalty=0.0, frequency_penalty=0.0, temperature=0.8, top_p=0.95, top_k=-1, use_beam_search=False, stop=[], ignore_eos=False, max_tokens=16, logprobs=None), num_seqs=1),
    SequenceGroup(request_id=1, sampling_params=SamplingParams(n=1, best_of=1, presence_penalty=0.0, frequency_penalty=0.0, temperature=0.8, top_p=0.95, top_k=-1, use_beam_search=False, stop=[], ignore_eos=False, max_tokens=16, logprobs=None), num_seqs=1),
    SequenceGroup(request_id=2, sampling_params=SamplingParams(n=1, best_of=1, presence_penalty=0.0, frequency_penalty=0.0, temperature=0.8, top_p=0.95, top_k=-1, use_beam_search=False, stop=[], ignore_eos=False, max_tokens=16, logprobs=None), num_seqs=1),
    SequenceGroup(request_id=3, sampling_params=SamplingParams(n=1, best_of=1, presence_penalty=0.0, frequency_penalty=0.0, temperature=0.8, top_p=0.95, top_k=-1, use_beam_search=False, stop=[], ignore_eos=False, max_tokens=16, logprobs=None), num_seqs=1)
]
```

第二步是调用 LLM._run_engine，通过 LLM.llm_engine.step()，转到 LLM.llm_engine._run_workers 函数中进行处理。

在 LLM.generate 的处理过程中，LLMEngine, Scheduler, Worker 协作配合，LLMEngine 负责总控，Scheduler 负责调度，Worker 负责执行。


### 4.4 vLLM 性能测试

推理系统常常需要被部署为在线服务,因此服务的稳定性、延迟、性能等指标的量化非常关键。vllm项目下benchmark目录提供了测试脚本供我们进行实验。

首先需要启动服务，与第一小节不同的是，脚本并不支持openai风格的接口：

```bash
python -m vllm.entrypoints.api_server --model /mlx/users/xingzheng.daniel/playground/model/chinese-alpaca-2-7b
```

然后运行脚本得到以下输出:
```bash
(torch2) ➜  benchmarks git:(main) python3 benchmark_serving.py --dataset ShareGPT_V3_unfiltered_cleaned_split.json --tokenizer  /mlx/users/xingzheng.daniel/playground/model/chinese-alpaca-2-7b --request-rate 40
Namespace(backend='vllm', host='localhost', port=8000, dataset='ShareGPT_V3_unfiltered_cleaned_split.json', tokenizer='/mlx/users/xingzheng.daniel/playground/model/chinese-alpaca-2-7b', best_of=1, use_beam_search=False, num_prompts=1000, request_rate=40.0, seed=0, trust_remote_code=False)
Total time: 165.50 s
Throughput: 6.04 requests/s
Average latency: 77.68 s
Average latency per token: 0.27 s
Average latency per output token: 1.03 s
输出的 token 总数 / 总时间: 1348.35 tokens/s
```

脚本逻辑总结来说，通过异步IO的方式发送N个网络请求，记录每一个单个请求所消耗的时长(latency), prompt token数量，output的token数量。最后根据这个数据计算一些统计指标。

vLLM 部署模型可能会消耗更多的显存，因为 vllm 会在初始化的时会预先分配大块内存（可以通过gpu_memory_utilization参数控制），这是因为之后所有内存分配都是在vllm内部进行，vllm接管了所有内存的分配与收回。




