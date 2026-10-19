# Ming-3.0-asr-tiny：SGLang 加速服务

[English](README.en.md) · [模型主页](../../README.md)

本指南介绍 SGLang 单卡服务的安装、启动与转写。普通转写使用 `/v1/audio/transcriptions`；数据集评测使用下文的 `/generate` 请求，两者的提示词和解码设置不同。

## 环境与版本

| 组件 | 版本 |
| --- | --- |
| 系统与 GPU | Linux x86-64，NVIDIA L20Y 80 GB，单卡 BF16 |
| Python | 3.12.3 |
| SGLang 公开上游 | `sgl-project/sglang`，提交 `81f27fb3a71d3504eab821a9d39352c924c730cf` |
| 模型适配 | 本仓库的 [三份补丁](patches/)；前两份加入 Ming ASR 音频支持，第三份匹配模型类名 |
| 模型 | `inclusionAI/Ming-3.0-asr-tiny`，提交 `d08d77d4b32f0707eb430a995ccbf933f0b04583` |
| 依赖 | [requirements.txt](requirements.txt) |

`requirements.txt` 提供服务依赖；`patches/` 包含两份 Ming ASR 音频适配补丁和一份模型类名补丁；`scripts/` 提供补丁应用、环境检查、启动及客户端示例。SGLang 适配补丁采用 [Apache-2.0 许可证](LICENSE-APACHE-2.0)。

## 1. 获取代码并创建环境

克隆本仓库后，在仓库根目录执行：

```bash
python3.12 -m venv .venv-sglang
source .venv-sglang/bin/activate
unset PYTHONPATH
python -m pip install -r cookbook/sglang/requirements.txt
python -m pip check
export GUIDE_DIR="$(pwd)/cookbook/sglang"
export PYTHON_BIN="$(command -v python)"
```

服务端需要 CUDA GPU。HTTP 客户端只需 Python 标准库，不需要 GPU 或安装上述依赖。不要把模型包的 HF `requirements.txt` 装进这个 SGLang 环境：两套推理路径使用不同的 Torch 依赖组合。

获取公开上游的固定版本；模型适配补丁已随本仓库发布，不需要另一个私有仓库：

```bash
git clone https://github.com/sgl-project/sglang.git ../sglang_ming_asr
export SGLANG_DIR="$(cd ../sglang_ming_asr && pwd)"
git -C "$SGLANG_DIR" checkout 81f27fb3a71d3504eab821a9d39352c924c730cf
```

若已有该提交的完整本地源码副本，也可以直接把 `SGLANG_DIR` 指向它。请用 `git -C "$SGLANG_DIR" rev-parse HEAD` 核对提交；普通 PyPI 版 SGLang 只提供依赖，运行时仍需加载打过补丁的源码。

## 2. 准备模型并应用补丁

下载固定模型版本，或使用内容相同的完整本地模型目录：

```bash
hf download inclusionAI/Ming-3.0-asr-tiny \
  --revision d08d77d4b32f0707eb430a995ccbf933f0b04583 \
  --local-dir ../Ming-3.0-asr-tiny-model
export MODEL_DIR="$(cd ../Ming-3.0-asr-tiny-model && pwd)"
```

如果模型已在本地，改为 `export MODEL_DIR=/absolute/path/to/model`。目录须包含配置、tokenizer、processor、权重索引和全部 safetensors 分片。先检查模型包：

```bash
python tools/verify_repository.py --checkpoint "$MODEL_DIR"
```

前两份补丁将 Ming ASR 音频支持加入固定上游，最后一份将模型的 `MingV3ASR` 类名映射到该音频实现。补丁只修改 SGLang 源码，不修改模型文件；适配代码沿用 SGLang 的 [Apache-2.0 许可证](LICENSE-APACHE-2.0)：

```bash
SGLANG_DIR="$SGLANG_DIR" bash "$GUIDE_DIR/scripts/apply_patch.sh"
export PYTHONPATH="$SGLANG_DIR/python:$MODEL_DIR"
"$PYTHON_BIN" "$GUIDE_DIR/scripts/check_env.py" \
  --model "$MODEL_DIR" --sglang-dir "$SGLANG_DIR"
```

`check_env.py` 会检查 GPU、BF16、权重分片和实际导入的源码路径。补丁可重复执行；脚本会识别已应用状态。

## 3. 启动服务

```bash
export GPU_ID=0 PORT=8000 HOST=127.0.0.1 SERVED_MODEL=ming-asr
PYTHON_BIN="$PYTHON_BIN" MODEL_DIR="$MODEL_DIR" SGLANG_DIR="$SGLANG_DIR" \
  bash "$GUIDE_DIR/scripts/start_server.sh"
```

服务默认使用单卡、BF16、TP=1。脚本启动前会检查端口是否可用；需要换端口时设置 `PORT`。首次启动可能编译 GPU 内核，等健康检查返回 200 后再请求：

```bash
curl -fsS --max-time 30 http://127.0.0.1:8000/health
curl -fsS --max-time 30 http://127.0.0.1:8000/v1/models
```

若从容器外访问，容器内设置 `HOST=0.0.0.0`，并在宿主机限制端口访问。这个示例服务没有对外鉴权，不应直接暴露在公网。

## 4. 转写音频

另开终端，进入模型仓库根目录。普通转写客户端只使用 Python 标准库：

```bash
python cookbook/sglang/scripts/transcribe.py \
  --audio examples/cases/audio/chinese_common_voice.wav \
  --base-url http://127.0.0.1:8000 --model ming-asr

python cookbook/sglang/scripts/transcribe.py \
  --audio examples/cases/audio/english_common_voice.wav \
  --base-url http://127.0.0.1:8000 --model ming-asr
```

两段仓库样例的识别结果分别是“漓江和湘江有秦代开凿的灵渠沟通”和“same way you did”。客户端也可以用 curl：

```bash
curl --fail-with-body http://127.0.0.1:8000/v1/audio/transcriptions \
  -F 'file=@examples/cases/audio/chinese_common_voice.wav' \
  -F 'model=ming-asr' -F 'response_format=json' -F 'temperature=0'
```

## 5. 数据集评测

公开数据集质量对照使用精确的 ASR 提示、greedy decoding、`max_new_tokens=256` 和 10-gram 重复限制。在客户端终端激活刚创建的环境，并将 `SGLANG_DIR` 指向第 1 节的适配源码：

```bash
source .venv-sglang/bin/activate
unset PYTHONPATH
export SGLANG_DIR="$(cd ../sglang_ming_asr && pwd)" # 若使用其他源码目录，请改为其绝对路径
python cookbook/sglang/scripts/transcribe_aligned.py \
  --sglang-dir "$SGLANG_DIR" \
  --audio examples/cases/audio/english_common_voice.wav \
  --base-url http://127.0.0.1:8000
```

该请求只适合可信评测客户端；普通应用使用标准转写接口。不要将标准接口的输出直接当作下表所用的评测协议。

| 数据集 | SGLang | HF Transformers | 样本数 |
| --- | ---: | ---: | ---: |
| AISHELL-1 CER | 1.058560% | 1.044242% | 7,176 |
| LibriSpeech test-clean WER | 1.169216% | 1.169216% | 2,620 |
| LibriSpeech test-other WER | 2.405355% | 2.394009% | 2,939 |

共 12,735 条。三组采用相同参考文本与归一化规则；表中 SGLang 结果来自上述 `/generate` 请求。

## 常见问题

- `Address already in use`：换一个空闲 `PORT`。启动脚本会在加载权重前提示。
- 健康检查首次返回 503：等待 GPU 内核编译完成后重试；以 HTTP 200 为就绪标准。
- `Wrong SGLang source imported`：清除旧 `PYTHONPATH`，重新设置为 `$SGLANG_DIR/python:$MODEL_DIR`。
- `MingV3ASR` 类名无法识别：核对源码提交并运行 `apply_patch.sh`。
- `torchcodec` 提示缺少系统 FFmpeg 库：WAV 文件可使用音频解码回退路径；其他音频格式需安装相应解码依赖。
