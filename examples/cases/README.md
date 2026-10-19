# Three Reproducible Audio Examples

The examples use the HF model and this repository's Transformers script with BF16, SDPA, greedy decoding, and `max_new_tokens=256` on a CUDA GPU. Ordinary transcription and Context ASR use the same input format. Both requests in the Context comparison seed known Chinese; only the second adds a topic and hotwords. Set up `MODEL_DIR` and dependencies as described in the [English quick start](../../README.md#quick-start).

Open each WAV link to listen to the sample.

## 1. Context ASR: Correcting a Film Title with Hotwords

[▶ Play audio](https://niniandy.github.io/Ming-3.0-asr-tiny-blog/assets/audio/context_same_sentence_pcm16.wav)

6.57 seconds · ContextASR-Bench ID `04385-3_0-105184_0`

| Item | Text |
| --- | --- |
| Reference | 你们看了黄国权那部**同囚**没我昨晚看完到现在还缓不过来 |
| Topic | 同囚 |
| Hotwords | 同囚、黄国权、游学修、关楚耀、赵永洪、赵咏瑶、狱警、长镜头、吞玻璃戏、糖化玻璃、真实案例数据 |
| Without context | 你们看了黄国权那部**铜球**没我昨晚看完到现在还缓不过来 |
| With context | 你们看了黄国权那部**同囚**没我昨晚看完到现在还缓不过来 |

The same audio is used twice; only the second request receives the topic and hotwords above.

```bash
python examples/transcribe.py \
  --model "$MODEL_DIR" \
  --manifest examples/cases/context.jsonl \
  --output outputs/context-demo.jsonl \
  --language-hint zh --max-new-tokens 256
```

## 2. English

[▶ Play audio](https://niniandy.github.io/Ming-3.0-asr-tiny-blog/assets/audio/english_common_voice.wav)

2.26 seconds · Common Voice 15 ID `common_voice_en_100540`

| Item | Text |
| --- | --- |
| Reference | same way you did |
| Model output | same way you did |

```bash
python examples/transcribe.py --model "$MODEL_DIR" \
  --audio examples/cases/audio/english_common_voice.wav \
  --max-new-tokens 256
```

## 3. Mandarin

[▶ Play audio](https://niniandy.github.io/Ming-3.0-asr-tiny-blog/assets/audio/chinese_common_voice.wav)

5.90 seconds · Common Voice 15 ID `common_voice_zh-CN_22099416`

| Item | Text |
| --- | --- |
| Reference | 漓江和湘江有秦代开凿的灵渠沟通 |
| Model output | 漓江和湘江有秦代开凿的灵渠沟通 |

```bash
python examples/transcribe.py --model "$MODEL_DIR" \
  --audio examples/cases/audio/chinese_common_voice.wav \
  --max-new-tokens 256
```

Audio sources: The Context sample comes from [ContextASR-Bench](https://huggingface.co/datasets/MrSupW/ContextASR-Bench), whose dataset page states MIT; see the included [license text](CONTEXTASR_BENCH_LICENSE.txt). The English and Mandarin samples come from [Mozilla Common Voice](https://commonvoice.mozilla.org/en/datasets) (CC0). The Context audio was converted to browser-compatible 16-bit PCM WAV without rerecording. These clips demonstrate their own recognition results, not a general Context ASR gain.
