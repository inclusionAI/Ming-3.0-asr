# Context ASR

[Model README](../README.md)

Context ASR and ordinary transcription use **one input format**. Append the topic and comma-separated hotwords directly to the base ASR instruction, with no newline, then append the audio. When the language is known, seed `Chinese\t` or `English\t` after the audio tokens. The chunk policy uses 3000 mel frames for clips up to 30 seconds and 6000 for clips from 30 to 60 seconds.

```bash
export MODEL_DIR=/path/to/Ming-3.0-asr-tiny
python examples/transcribe.py \
  --model "$MODEL_DIR" --audio /path/to/speech.wav \
  --language-hint zh --context-chunks \
  --prompt $'Please recognize the language of this speech and transcribe it, placing a `\t` between the identified language and the transcribe.This is an audio about 电影蜥蜴.This audio may contains the following words or phrases:蜥蜴,姜知垠,曹承佑,姜惠贞,八音盒.'
```

Use `--language-hint en` for known English audio, or omit the hint if the language is unknown. Keep the same language hint, chunk policy, BF16, SDPA, greedy decoding, `max_new_tokens=256`, and `no_repeat_ngram_size=10` when comparing with and without context; change only the prompt. Dialogue history can also be included as text in the prompt. The example words are illustrative.

## Python

After loading `processor`, `model`, `waveform`, and `rate` as shown in [Transformers inference](hf-inference.md):

```python
import torch
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

Hotwords are hints, not guaranteed output. The processor has no separate `history` argument; include dialogue history in the text prompt.
