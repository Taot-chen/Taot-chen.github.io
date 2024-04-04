---
layout: post
title: deploys_LLM_vLLM-vs-TGI
date: 2024-04-05
tags: [llm]
author: taot
---


## vLLM vs TGI 部署大模型以及注意点


LLM 高并发部署是个难题，具备高吞吐量的服务，能够让用户有更好的体验（比如模型生成文字速度提升，用户排队时间缩短）。

[vllm github 仓库](https://github.com/vllm-project/vllm/tree/main)


### 1 vLLM

#### 1.1 启动模型服务
  
```bash
# cd /workspace/vllm
python3 -m vllm.entrypoints.api_server \
    --model /models/Llama-2-7b-chat-hf --swap-space 16 \
    --disable-log-requests --host 0.0.0.0 --port 8080 --max-num-seqs 256
```

更多参数可以在 vllm/engine/arg_utils.py 找到，其中比较重要的有：

* max-num-seqs：默认 256，当 max-num-seqs 比较小时，较迟接收到的 request 会进入 waiting_list，直到前面有request 结束后再被添加进生成队列。当 max-num-seqs 太大时，会出现一部分 request 在生成了 3-4 个 tokens 之后，被加入到 waiting_list（有些用户出现生成到一半卡住的情况）。过大或过小的 max-num-seqs 都会影响用户体验。
* max-num-batched-tokens：很重要的配置，比如你配置了 max-num-batched-tokens=1000 那么你大概能在一个 batch 里面处理 10 条平均长度约为 100 tokens 的 inputs。max-num-batched-tokens 应尽可能大，来充分发挥 continuous batching 的优势。不过似乎（对于 TGI 是这样，vllm 不太确定），在提供 HF 模型时，该 max-num-batched-tokens 能够被自动推导出来。

部署后，发送 post 请求到 http://{host}:{port}/generate ，body 为 ：

```bash
{
    "prompt": "Once a upon time,",
    "max_tokens": output_len,
    "stream": false,
    "top_p": 1.0
    // 其他参数
}
```

vllm 也提供了 OpenAI-compatible API server，vllm 调用了 fastchat 的 conversation template 来构建对话输入的 prompt，但 v0.1.2 的 vllm 与最新版的 fastchat 有冲突，为了保证使用 llama v2 时用上对应的 prompt template，可以手动修改以下entrypoints.openai.api_server 中对 get_conversation_template 的引入方式（[link](https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/openai/api_server.py#L33)）。修改后执行以下代码启动： 

```bash
python3 -m vllm.entrypoints.openai.api_server \
        --model /models/Llama-2-7b-chat-hf \
        --disable-log-requests --host 0.0.0.0 --port 5001 --max-num-seqs 20 --served-model-name llama-2
```

#### 1.2 坑点

* 采用 stream: true 进行 steaming 请求时，vllm 提供的默认 API 每次的回复并不是新预测的单词，而是目前所有已预测的全部文本内容（history_tokens + new_tokens），这导致回复的内容中包含很多冗余的信息。
* 当前 vllm 版本（v0.1.2）加载 *.safetensor 模型时存在问题，请尽量加载 *.bin 格式的模型。对于文件夹下同时存放 bin 和 safetensor 权重时，vllm 优先加载 .bin 权重。
* vllm 的 OpenAI-compatible API server 依赖 fschat 提供 prompt template，由于 LLM 更新进度快，如果遇到模型 prompt template 在 fschat 中未找到的情况（通常报错为 keyError），可以重新安装下 fschat 和 transformers。


从接收 request，到返回回复，大致的过程如下：

* 1 vllm 接收到 request 之后，会发放 request_uuid，并将 request 分配到 running, swap, waiting 三个队列当中。（参考 vllm.core.scheduler.Scheduler._schedule ）
* 2 根据用户等待的时间进行 request 优先级排序。从 running 和 swap队列中选择优先级高的 request 来生成对应的回复，由于 decoding 阶段，每次前项传播只预测一个 token，因此 vllm 在进行完一次前项传播（即 one decoding iteration）之后，会返回所有新生成的 tokens 保存在每个 request_uuid 下。（参考 vllm.engine.llm_engine.LLMEngine.step）
* 3 如果 request 完成了所有的 decoding 步骤，那么将其移除，并返回结果给用户。
* 4 更新 running, swap 和 waiting 的 request。
* 5 循环执行 2,3,4。


### 2 TGI(Text Generation Inference)

[TGI github 仓库](https://github.com/huggingface/text-generation-inference)

TGI 特点：

* 支持了和 vllm 类似的 continuous batching
* 支持了 flash-attention 和 Paged Attention。
* 支持了 Safetensors 权重加载。（目前版本的 vllm 在加载部分模型的 safetensors 有问题（比如 llama-2-7B-chat）。
* TGI 支持部署 GPTQ 模型服务，这使得我们可以在单卡上部署拥有 continous batching 功能的，更大的模型。
* 支持采用 Tensor Parallelism 部署多 GPU 服务，模型水印等其他功能

#### 2.1 安装

如果想直接部署模型的话，建议通过 docker 安装，省去不必要的环境问题。目前 TGI 只提供了 0.9.3 的：

```bash
docker pull ghcr.io/huggingface/text-generation-inference:0.9.3
```

如果要进行本地测试，可以通过源码安装（以下在 ubuntu 上安装）：

* 依赖安装

```bash
# 如果没有网络加速的话，建议添加 pip 清华源或其他国内 pip 源
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
apt-get install cargo  pkg-config git
```

* 下载 protoc

```bash
PROTOC_ZIP=protoc-21.12-linux-x86_64.zip
curl -OL https://github.com/protocolbuffers/protobuf/releases/download/v21.12/$PROTOC_ZIP
sudo unzip -o $PROTOC_ZIP -d /usr/local bin/protoc
sudo unzip -o $PROTOC_ZIP -d /usr/local 'include/*'
rm -f $PROTOC_ZIP
```

* 如果没有网络加速的话，建议修改 cargo 源。有网络加速可略过。

```bash
# vim ~/.cargo/config
[source.crates-io]
registry = "https://github.com/rust-lang/crates.io-index"
​
replace-with = 'tuna'
​
[source.tuna]
registry = "https://mirrors.tuna.tsinghua.edu.cn/git/crates.io-index.git"
​
[net]
git-fetch-with-cli=true
```

* TGI 根目录下执行安装

```bash
BUILD_EXTENSIONS=True make install # Install repository and HF/transformer fork with CUDA kernels
```

* 安装成功，添加环境变量到 .bashrc 中 export PATH=/root/.cargo/bin:$PATH

* 执行text-generation-launcher --help ，有输出表示安装成功。

#### 2.2 使用

将 Llama-2-7b-chat-hf 部署成服务：

```bash
# 建议将模型下载到本地，而后挂载到 docker 中，避免在 docker 中重复下载。
docker run --rm \
	--name tgi \
	--runtime=nvidia \
	--gpus all \
	-p 5001:5001 \
	-v /home/kevin/models:/data \
	ghcr.io/huggingface/text-generation-inference:0.9.3 \
	--model-id /data/Llama-2-7b-chat-hf \
	--hostname 0.0.0.0 \
	--port 5001 \
	--dtype float16 \
	--sharded false 
```

可以通过 text-generation-launcher --help 查看到可配置参数，相对 vllm 来说，TGI 在服务部署上的参数配置更丰富一些，其中比较重要的有：

* model-id：模型 path 或者 hf.co 的 model_id。
* revision：模型版本，比如 hf.co 里仓库的 branch名称。
* quantize：TGI 支持使用 GPTQ 来部署模型。
* max-concurrent-requests：当服务处于高峰期时，对于更多新的请求，系统会直接返回拒绝请求，（比如返回服务器正忙，请稍后重试），而不是将新请求加入到 waiting list 当中。改配置可以有效环节后台服务压力。默认为 128。
* max-total-tokens：相当于模型的 max-tokens
* max-batch-total-tokens：非常重要的参数，他极大影响了服务的吞吐量。该参数与 vllm 中的 max-num-batched-tokens 类似。比如你配置了 max-num-batched-tokens=1000：那么你大概能在一个 batch 里面处理 10 条平均长度约为 100 tokens 的 inputs。对于传入的 HF 模型，TGI 会自动推理该参数的最大上限，如果你加载了一个 7B 的模型到 24GB 显存的显卡当中，你会看到你的显存占用基本上被用满了，而不是只占用了 13GB（7B 模型常见显存占用），那是因为 TGI 根据 max-batch-total-tokens 提前对显存进行规划和占用。但对于量化模型，该参数需要自己设定，设定时可以根据显存占用情况，推测改参数的上限。

