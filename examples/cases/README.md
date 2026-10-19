# 三个可复现的音频样例 · Three reproducible audio examples

这些样例使用 [HF 模型包](https://huggingface.co/inclusionAI/Ming-3.0-asr-tiny)和本仓库的 Transformers 示例，在 CUDA GPU 上以 BF16、SDPA、greedy decoding 和 `max_new_tokens=256` 运行。普通转写与 Context ASR 均采用同一输入格式；下方 Context 对比的两次请求都预填已知中文语种，仅后一次增加主题和热词。展示文本统一去除标点，英文转为小写，便于比较。

The examples use the HF model and this repository's Transformers script with BF16, SDPA, greedy decoding, and `max_new_tokens=256` on a CUDA GPU. Ordinary transcription and Context ASR use the same input format. Both requests in the Context comparison seed known Chinese; only the second adds a topic and hotwords. Set up `MODEL_DIR` and dependencies as described in the [English quick start](../../README.md#quick-start).

播放器如默认静音，请点击音量按钮。If the player starts muted, use its speaker control to enable sound.

## 1. Context ASR：同一音频，热词纠正片名

https://github.com/user-attachments/assets/62613e2e-6dfa-4f1d-aaa1-ab83dcd78aed

6.57 秒 · ContextASR-Bench ID `04385-3_0-105184_0`

| 项目 / Item | 文本 / Text |
| --- | --- |
| 参考文本 / Reference | 你们看了黄国权那部**同囚**没我昨晚看完到现在还缓不过来 |
| 主题 / Topic | 同囚 |
| 热词 / Hotwords | 同囚、黄国权、游学修、关楚耀、赵永洪、赵咏瑶、狱警、长镜头、吞玻璃戏、糖化玻璃、真实案例数据 |
| 不加上下文 / Without context | 你们看了黄国权那部**铜球**没我昨晚看完到现在还缓不过来 |
| 加入主题与热词 / With context | 你们看了黄国权那部**同囚**没我昨晚看完到现在还缓不过来 |

两次使用完全相同的音频，仅后一次加入以上主题与热词。The same audio is used twice; only the second request receives the topic and hotwords above.

```bash
python examples/transcribe.py \
  --model "$MODEL_DIR" \
  --manifest examples/cases/context.jsonl \
  --output outputs/context-demo.jsonl \
  --language-hint zh --max-new-tokens 256
```

## 2. 英语 / English

https://github.com/user-attachments/assets/620afa11-e8d9-4550-ae64-809c36dfcd8b

2.26 秒 · Common Voice 15 ID `common_voice_en_100540`

| 项目 / Item | 文本 / Text |
| --- | --- |
| 参考文本 / Reference | same way you did |
| 识别结果 / Output | same way you did |

```bash
python examples/transcribe.py --model "$MODEL_DIR" \
  --audio examples/cases/audio/english_common_voice.wav \
  --max-new-tokens 256
```

## 3. 中文 / Mandarin

https://github.com/user-attachments/assets/a13297d3-1458-406d-8846-585d0cbf37b8

5.90 秒 · Common Voice 15 ID `common_voice_zh-CN_22099416`

| 项目 / Item | 文本 / Text |
| --- | --- |
| 参考文本 / Reference | 漓江和湘江有秦代开凿的灵渠沟通 |
| 识别结果 / Output | 漓江和湘江有秦代开凿的灵渠沟通 |

```bash
python examples/transcribe.py --model "$MODEL_DIR" \
  --audio examples/cases/audio/chinese_common_voice.wav \
  --max-new-tokens 256
```

音频来源 / Audio sources: Context 样例来自 [ContextASR-Bench](https://huggingface.co/datasets/MrSupW/ContextASR-Bench)（其数据集页面标注 MIT，许可文字见[这里](CONTEXTASR_BENCH_LICENSE.txt)）；中英文样例来自 [Mozilla Common Voice](https://commonvoice.mozilla.org/en/datasets)（CC0）。Context 音频仅转换为浏览器兼容的 16-bit PCM WAV，未重配音。Context 样例只展示这条音频的实体纠错，不代表对全部热词或长音频都有效。

The ContextASR-Bench dataset page marks the dataset MIT-licensed; Mozilla distributes Common Voice under CC0. This demonstration establishes only the behavior of these clips, not a general context-ASR gain.
