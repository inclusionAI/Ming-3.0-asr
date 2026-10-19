# Context ASR

[English](context-asr.en.md) · [返回中文首页](../README.md)

Context ASR 利用热词列表或对话上文增强领域实体识别，例如人名、地名、专业术语和产品名称。

## 自定义转写提示

Context ASR 与普通转写使用**同一输入格式**：基础识别指令后连续拼接主题和热词，最后接音频。已知语种时，在音频 token 后预填 `Chinese\t` / `English\t`。下面的命令会按音频时长选择分块：不超过 30 秒用 3000 mel 帧，30–60 秒用 6000 mel 帧。

```bash
export MODEL_DIR=/path/to/Ming-3.0-asr-tiny
python examples/transcribe.py \
  --model "$MODEL_DIR" --audio /path/to/speech.wav \
  --language-hint zh --context-chunks \
  --prompt $'Please recognize the language of this speech and transcribe it, placing a `\t` between the identified language and the transcribe.This is an audio about 电影蜥蜴.This audio may contains the following words or phrases:蜥蜴,姜知垠,曹承佑,姜惠贞,八音盒.'
```

主题与热词之间不插入换行；示例中的词仅供演示。英文音频将 `--language-hint zh` 改为 `en`；语种未知时可不传语言前缀。比较有无上下文时，两次都使用相同的语言前缀、分块和解码设置，只改变提示内容。对话上文也可作为文本提示的一部分传入。

## Python 接口

在 [HF 推理示例](hf-inference.md)加载模型和波形后，可以替换输入准备部分：

```python
from examples.hf_runtime import (
    ASR_INSTRUCTION, context_chunk_frames,
    decode_transcript, prepare_asr_request,
)

prompt = (
    ASR_INSTRUCTION +
    "This is an audio about 电影蜥蜴."
    "This audio may contains the following words or phrases:"
    "蜥蜴,姜知垠,曹承佑,姜惠贞,八音盒."
)
chunk_frames = context_chunk_frames(len(waveform) / rate)
model.config.audio_config.chunk_frames = chunk_frames
inputs = prepare_asr_request(
    processor, (waveform.T, rate), prompt=prompt,
    language_hint="zh", audio_chunk_frames=chunk_frames,
).to(model.device)
with torch.inference_mode():
    eos = model.config.text_config.eos_token_id
    output = model.generate(
        **inputs, max_new_tokens=256, do_sample=False,
        eos_token_id=eos, pad_token_id=eos, no_repeat_ngram_size=10,
    )
tokens = output[0, inputs.input_ids.shape[1]:].tolist()
print(decode_transcript(processor.tokenizer, tokens))
```

## 使用建议

- 选择与当前音频相关的热词，避免加入过多无关实体。
- 上下文和音频共同占用模型上下文窗口，应控制提示长度。
- 对话上文需要组织为文本提示；当前 processor 没有独立的 `history` 参数。
- 热词是识别提示，不保证输出必然包含对应词语。
- 评估时同时关注普通转写错误率和实体识别指标，避免只衡量热词命中。
- Context ASR 与普通转写共用 `decode_transcript`；函数会移除输出中的语言标签，只保留转写文本。

[查看 Context ASR 评测图](../README.md#context-asr) · [返回 cookbook](README.md)
