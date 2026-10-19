# Local Transcription with Transformers

This repository uses the custom-code model package on Hugging Face. Create a separate Python environment and download the complete model package. Its requirements include PyTorch and torchaudio. For another CUDA version, first install matching 2.11.0 builds using the [official PyTorch instructions](https://pytorch.org/get-started/previous-versions/).

```bash
export MODEL_DIR=/path/to/Ming-3.0-asr-tiny
python -m pip install -r "$MODEL_DIR/requirements.txt"
python tools/verify_repository.py --checkpoint "$MODEL_DIR"
python examples/transcribe.py --model "$MODEL_DIR" --audio /path/to/speech.wav
```

The runtime uses Python 3.12, Linux, Transformers 5.12.1, fla-core 0.5.2, and PyTorch 2.11.0+cu130. The model directory must contain the source files, configuration, tokenizer, weight index, and every weight shard.

## Save Output and Select a GPU

```bash
python examples/transcribe.py \
  --model "$MODEL_DIR" --audio /path/to/speech.wav \
  --device cuda:0 --attention sdpa --max-new-tokens 256 \
  --output outputs/transcript.json
```

The output includes generated token IDs, transcript, model path, and audio path. Decoding is greedy by default. If generation reaches the token limit, the transcript may be truncated.

## Python API

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

## Input and Precision

- `soundfile` returns sample-first audio. Pass `waveform.T` to the processor for channel-first input, together with the original sample rate.
- The frontend downmixes and resamples to 16 kHz. Chunked audio encoding does not make the input length unlimited; input and output share the 8192-token context limit.
- Use `from_pretrained(..., dtype=torch.bfloat16)`. Do not call `.bfloat16()` on the entire model afterward, since some routing correction buffers must remain FP32.
- Ordinary transcription and Context ASR both use `prepare_asr_request`. Context text is appended immediately after the same base instruction and before `<AUDIO>`; see [Context ASR](context-asr.en.md).
- The checkpoint configuration controls audio chunks by default. For Context ASR, use 3000 mel frames up to 30 seconds or 6000 for 30–60 seconds; `--context-chunks` selects this automatically.
- Decoding removes the language field at the tab separator. Without a tab, the evaluation-compatible fallback strips leading `Chinese` or `English` labels.
- `--prompt` supplies context. For a known language, `--language-hint zh|en` seeds the language after the audio tokens. `--audio-chunk-frames` overrides chunking but cannot be combined with `--context-chunks`. The Context ASR example uses BF16, SDPA, greedy decoding, `max_new_tokens=256`, and `no_repeat_ngram_size=10`; see [Context ASR](context-asr.en.md).

[Back to Cookbook](README.md)
