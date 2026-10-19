# Transformers 本地转写

本仓库调用 HF custom-code 模型包，不再镜像模型源码或配置。先创建独立 Python 环境，并从 HF 下载完整模型；下面的依赖文件包含 PyTorch 和 torchaudio。如需其他 CUDA 版本，请先按 [PyTorch 官方说明](https://pytorch.org/get-started/previous-versions/)安装匹配的 2.11.0 构建。

```bash
export MODEL_DIR=/path/to/Ming-3.0-asr-tiny
python -m pip install -r "$MODEL_DIR/requirements.txt"
python tools/verify_repository.py --checkpoint "$MODEL_DIR"
python examples/transcribe.py --model "$MODEL_DIR" --audio /path/to/speech.wav
```

运行环境为 Python 3.12、Linux、Transformers 5.12.1、fla-core 0.5.2、PyTorch 2.11.0+cu130。模型目录应包含源码、配置、tokenizer、权重索引及全部分片。

## 保存结果与选择 GPU

```bash
python examples/transcribe.py \
  --model "$MODEL_DIR" --audio /path/to/speech.wav \
  --device cuda:0 --attention sdpa --max-new-tokens 256 \
  --output outputs/transcript.json
```

输出包含原始生成 token IDs、转写文本、模型和音频路径。默认 greedy decoding；达到生成长度上限可能截断转写，不能把截断结果当作完整文本。

## Python 调用

```python
import soundfile as sf
import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from examples.hf_runtime import decode_transcript, prepare_asr_request

checkpoint = "/path/to/complete/hf-checkpoint"
processor = AutoProcessor.from_pretrained(checkpoint, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    checkpoint, trust_remote_code=True, dtype=torch.bfloat16,
    device_map="cuda:0", attn_implementation="sdpa",
).eval()
waveform, rate = sf.read("speech.wav", dtype="float32", always_2d=True)
inputs = prepare_asr_request(processor, (waveform.T, rate)).to(model.device)
with torch.inference_mode():
    eos = model.config.text_config.eos_token_id
    output = model.generate(
        **inputs, max_new_tokens=256, do_sample=False,
        eos_token_id=eos, pad_token_id=eos, no_repeat_ngram_size=10,
    )
print(decode_transcript(
    processor.tokenizer, output[0, inputs.input_ids.shape[1]:].tolist(),
))
```

## 输入与精度

- `soundfile` 返回 sample-first 波形，传入 processor 前用 `waveform.T` 转为 channel-first，并携带原采样率。
- 前端混音并重采样到 16 kHz；分块编码不等于支持无限长音频，总输入和输出仍受 8192 token 上下文限制。
- 使用 `from_pretrained(..., dtype=torch.bfloat16)`，不要再对模型整体调用 `.bfloat16()`，避免把 FP32 路由校正等缓冲区降精度。
- 普通转写和 Context ASR 都使用 `prepare_asr_request`；Context ASR 在相同的基础指令后、`<AUDIO>` 前连续拼接提示内容。用法见 [Context ASR](context-asr.md)。
- 默认音频分块由 checkpoint 配置决定。Context ASR 对不超过 30 秒的音频使用 3000 mel 帧，30–60 秒使用 6000 mel 帧；命令行可用 `--context-chunks` 自动选择。
- 转写结果优先按制表符去掉语言字段；若无制表符，沿用评测规则去掉开头的 `Chinese`、`English` 等语言前缀。
- `--prompt` 可提供上下文；已知语言时，`--language-hint zh|en` 会在音频 token 后预填语言。`--audio-chunk-frames` 可覆盖分块，但不能与 `--context-chunks` 同时使用。Context ASR 示例采用 BF16、SDPA、greedy、`max_new_tokens=256` 和 `no_repeat_ngram_size=10`；具体提示见 [Context ASR](context-asr.md)。

[返回 cookbook](README.md)
