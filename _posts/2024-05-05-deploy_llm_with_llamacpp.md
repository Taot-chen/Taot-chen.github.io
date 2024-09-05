---
layout: post
title: deploy_llm_with_llamacpp
date: 2024-05-05
tags: [llm]
author: taot
---

## 使用llama.cpp量化部署LLM

以llama.cpp工具为例，介绍模型量化并在本地部署的详细步骤。这里使用 Meta最新开源的 Llama3-8B 模型。

### 1 环境
* 系统应有`make`（MacOS/Linux自带）或`cmake`（Windows需自行安装）编译工具
* Python 3.10以上编译和运行该工具


### 2 克隆和编译llama.cpp

拉取 llama.cpp 仓库最新代码
```bash
git clone https://github.com/ggerganov/llama.cpp.git
```

对llama.cpp项目进行编译，生成`./main`（用于推理）和`./quantize`（用于量化）二进制文件。
```bash
make
```

Windows/Linux用户如需启用GPU推理，则推荐[与BLAS（或cuBLAS如果有GPU）一起编译](https://github.com/ggerganov/llama.cpp#blas-build)，可以提高prompt处理速度。以下是和`cuBLAS`一起编译的命令，适用于NVIDIA相关GPU。参考：[llama.cpp#blas-build](https://github.com/ggerganov/llama.cpp#blas-build)
```bash
make LLAMA_CUBLAS=1
```

macOS用户无需额外操作，llama.cpp已对ARM NEON做优化，并且已自动启用BLAS。M系列芯片推荐使用Metal启用GPU推理，显著提升速度。只需将编译命令改为：`LLAMA_METAL=1 make`，参考[llama.cpp#metal-build](https://github.com/ggerganov/llama.cpp#metal-build)
```bash
LLAMA_METAL=1 make
```

### 3 生成量化版本模型

目前`llama.cpp`已支持`.pth`文件以及huggingface格式`.bin`的转换。将完整模型权重转换为GGML的FP16格式，生成文件路径为`Meta-Llama-3-8B-hf/ggml-model-f32.gguf`。进一步对FP32模型进行`4-bit`量化，生成量化模型文件路径为`Meta-Llama-3-8B-hf/ggml-model-q4_0.gguf`
```
python convert.py Meta-Llama-3-8B-hf/ --vocab-type bpe
./quantize ./Meta-Llama-3-8B-hf/ggml-model-f16.gguf ./Meta-Llama-3-8B-hf/ggml-model-q4_0.gguf q4_0
```

* 可以-h 查看脚本的一些超参数

### 4 加载并启动模型

#### 4.1 CPU 推理

运行./main二进制文件，-m命令指定 Q4量化模型（也可加载ggml-FP16的模型）。
```bash
# run the inference 推理
./main -m ./Meta-Llama-3-8B-hf/ggml-model-q4_0.gguf -n 128
./main -m ./Meta-Llama-3-8B-hf/ggml-model-f16.gguf -n 128

#以交互式对话
./main -m ./Meta-Llama-3-8B-hf/ggml-model-q4_0.gguf --color -f prompts/alpaca.txt -ins -c 2048 --temp 0.2 -n 256 --repeat_penalty 1.3
#chat with bob
./main -m ./Meta-Llama-3-8B-hf/ggml-model-q4_0.gguf -n 256 --repeat_penalty 1.0 --color -i -r "User:" -f prompts/chat-with-bob.txt
```

* 如果想用GPU加速，放在GPU上，需要更换编译 llama.cpp 的方式
  > GPU推理：通过Metal编译则只需在./main中指定-ngl 1；cuBLAS编译需要指定offload层数，例如-ngl 40表示offload 40层模型参数到GPU
  > 在支持 Metal 的情况下，可以使用 --gpu-layers|-ngl 命令行参数启用 GPU 推理。任何大于 0 的值都会将计算卸载到 GPU

* 比较重要的参数：
  * -ins 启动类ChatGPT的对话交流模式
  * -f 指定prompt模板，alpaca模型请加载prompts/alpaca.txt 指令模板
  * -c 控制上下文的长度，值越大越能参考更长的对话历史（默认：512）
  * -n 控制回复生成的最大长度（默认：128）
  * --repeat_penalty 控制生成回复中对重复文本的惩罚力度
  * --temp 温度系数，值越低回复的随机性越小，反之越大
  * --top_p, top_k 控制解码采样的相关参数
  * -b 控制batch size（默认：512）
  * -t 控制线程数量（默认：8），可适当增加

### 5 API 方式调用, 架设server

`llama.cpp`还提供架设`server`的功能，用于API调用、架设简易`demo`等用途。

运行以下命令启动`server`，二进制文件`./server`在`llama.cpp`根目录，服务默认监听`127.0.0.1:8080`。这里指定模型路径、上下文窗口大小。如果需要使用GPU解码，也可指定`-ngl`参数。
```bash
./server -m ./Meta-Llama-3-8B-hf/ggml-model-q4_0.gguf -c 4096 -ngl 999
```

服务启动后，即可通过多种方式进行调用，例如利用curl命令。以下是一个示例脚本（同时存放在`scripts/llamacpp/server_curl_example.sh`），将Alpaca-2的模板进行包装并利用curl命令进行API访问。
```bash
# server_curl_example.sh

SYSTEM_PROMPT='You are a helpful assistant. 你是一个乐于助人的助手。'
# SYSTEM_PROMPT='You are a helpful assistant. 你是一个乐于助人的助手。请你提供专业、有逻辑、内容真实、有价值的详细回复。' # Try this one, if you prefer longer response.
INSTRUCTION=$1
ALL_PROMPT="[INST] <<SYS>>\n$SYSTEM_PROMPT\n<</SYS>>\n\n$INSTRUCTION [/INST]"
CURL_DATA="{\"prompt\": \"$ALL_PROMPT\",\"n_predict\": 128}"

curl --request POST \
    --url http://localhost:8080/completion \
    --header "Content-Type: application/json" \
    --data "$CURL_DATA"
```

给出一个示例指令。
```bash
bash server_curl_example.sh '请列举5条文明乘车的建议'
```

稍后返回响应结果。
```bash
{
	"content": " 以下是五个文明乘车的建议：1）注意礼貌待人，不要大声喧哗或使用不雅用语；2）保持车厢整洁卫生，丢弃垃圾时要及时处理；3）不影响他人休息和正常工作时间，避免在车厢内做剧烈运动、吃零食等有异味的行为；4）遵守乘车纪律，尊重公共交通工具的规则和制度；5）若遇到突发状况或紧急情况，请保持冷静并配合相关部门的工作。这些建议旨在提高公民道德水平和社会文明程度，共同营造一个和谐、舒适的乘坐环境。",
	"generation_settings": 
    {
		"frequency_penalty": 0.0,
		"ignore_eos": false,
		"logit_bias": [],
		"mirostat": 0,
		"mirostat_eta": 0.10000000149011612,
		"mirostat_tau": 5.0,
		"model": "zh-alpaca2-models/7b/ggml-model-q6_k.gguf",
		"n_ctx": 4096,
		"n_keep": 0,
		"n_predict": 128,
		"n_probs": 0,
		"penalize_nl": true,
		"presence_penalty": 0.0,
		"repeat_last_n": 64,
		"repeat_penalty": 1.100000023841858,
		"seed": 4294967295,
		"stop": [],
		"stream": false,
		"temp": 0.800000011920929,
		"tfs_z": 1.0,
		"top_k": 40,
		"top_p": 0.949999988079071,
		"typical_p": 1.0
	},
	"model": "zh-alpaca2-models/7b/ggml-model-q6_k.gguf",
	"prompt": " [INST] <<SYS>>\nYou are a helpful assistant. 你是一个乐于助人的助手。\n<</SYS>>\n\n请列举5条文明乘车的建议 [/INST]",
	"stop": true,
	"stopped_eos": true,
	"stopped_limit": false,
	"stopped_word": false,
	"stopping_word": "",
	"timings": 
    {
		"predicted_ms": 3386.748,
		"predicted_n": 120,
		"predicted_per_second": 35.432219934875576,
		"predicted_per_token_ms": 28.2229,
		"prompt_ms": 0.0,
		"prompt_n": 120,
		"prompt_per_second": null,
		"prompt_per_token_ms": 0.0
	},
	"tokens_cached": 162,
	"tokens_evaluated": 43,
	"tokens_predicted": 120,
	"truncated": false
}
```
