# Ming-3.0-asr-tiny: SGLang Serving

[Model README](../../README.md)

This guide covers single-GPU SGLang serving for Ming-3.0-asr-tiny. Use the [Ming audio adapter branch](https://github.com/inclusionAI/sglang_ling_v3/tree/ling_v3_audio_for_pr) at the pinned commit below, together with the released checkpoint and the model-class alias patch included here. Ordinary transcription and dataset evaluation use different prompts and decoding settings.

## Environment and versions

| Component | Version |
| --- | --- |
| Ming ASR adapter source | [`inclusionAI/sglang_ling_v3`, branch `ling_v3_audio_for_pr`](https://github.com/inclusionAI/sglang_ling_v3/tree/ling_v3_audio_for_pr), commit `bba862771a9e3413a062dad33e8b494ea9f7755b` |
| Model-class alias | [Bundled patch](patches/ming-v3-asr-alias.patch) maps the released model class |
| Checkpoint | `inclusionAI/Ming-3.0-asr-tiny`, audio-only revision `d08d77d4b32f0707eb430a995ccbf933f0b04583` |
| Compatibility patch | [`ming-v3-asr-alias.patch`](patches/ming-v3-asr-alias.patch) |
| Runtime | Linux x86-64, Python 3.12.3, NVIDIA L20Y 80 GB, BF16, one GPU |
| Dependencies | [requirements.txt](requirements.txt) |

This directory includes pinned dependencies (`requirements.txt`), a model-class alias patch, compatibility patches for the upstream base (`patches/`), and setup/server/client scripts (`scripts/`). The adapter patches retain the upstream [Apache-2.0 license](LICENSE-APACHE-2.0).

## 1. Clone the repo and install the serving environment

On a Linux CUDA host, clone this repository and run the following from its root:

```bash
python3.12 -m venv .venv-sglang
source .venv-sglang/bin/activate
unset PYTHONPATH
python -m pip install -r cookbook/sglang/requirements.txt
python -m pip check
export GUIDE_DIR="$(pwd)/cookbook/sglang"
export PYTHON_BIN="$(command -v python)"
```

Do not install the checkpoint's separate HF `requirements.txt` into this SGLang environment. The HTTP client needs only Python's standard library; only the server requires the CUDA environment.

## 2. Prepare the adapter and checkpoint

Clone the Ming ASR adapter branch and pin the validated source commit:

```bash
git clone --single-branch --branch ling_v3_audio_for_pr \
  https://github.com/inclusionAI/sglang_ling_v3.git ../sglang_ming_asr
export SGLANG_DIR="$(cd ../sglang_ming_asr && pwd)"
git -C "$SGLANG_DIR" checkout bba862771a9e3413a062dad33e8b494ea9f7755b
```

An existing full checkout of the same commit can also be used by setting `SGLANG_DIR` to its absolute path. Check it with `git -C "$SGLANG_DIR" rev-parse HEAD`. The PyPI SGLang package supplies dependencies; runtime code comes from this adapter checkout.

Download the checkpoint, or use a complete local copy of the same revision:

```bash
hf download inclusionAI/Ming-3.0-asr-tiny \
  --revision d08d77d4b32f0707eb430a995ccbf933f0b04583 \
  --local-dir ../Ming-3.0-asr-tiny-model
export MODEL_DIR="$(cd ../Ming-3.0-asr-tiny-model && pwd)"
```

For an existing copy, set `MODEL_DIR=/absolute/path/to/model`. The directory must contain both safetensors shards, the index, configuration, tokenizer, processor, and custom Python files. Check it with `python tools/verify_repository.py --checkpoint "$MODEL_DIR"`.

## 3. Apply the class-name mapping and preflight

The adapter branch already contains Ming ASR audio support. The script applies only the `MingV3ASR` class-name alias to this commit; it does not change checkpoint weights or generation settings. The compatibility patches follow SGLang's [Apache-2.0 license](LICENSE-APACHE-2.0):

```bash
SGLANG_DIR="$SGLANG_DIR" bash "$GUIDE_DIR/scripts/apply_patch.sh"
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$SGLANG_DIR/python:$MODEL_DIR"
"$PYTHON_BIN" "$GUIDE_DIR/scripts/check_env.py" \
  --model "$MODEL_DIR" --sglang-dir "$SGLANG_DIR"
```

The preflight checks file presence, the actual imported adapter source, CUDA/BF16, and resolved architecture.

## 4. Start the server

Run in the foreground first, using a free GPU:

```bash
export GPU_ID=0 PORT=8000 HOST=127.0.0.1 SERVED_MODEL=ming-asr
PYTHON_BIN="$PYTHON_BIN" MODEL_DIR="$MODEL_DIR" SGLANG_DIR="$SGLANG_DIR" \
  bash "$GUIDE_DIR/scripts/start_server.sh"
```

In a Docker container accessed through a published port, set `HOST=0.0.0.0` inside the container, while restricting the host-side port as appropriate. The script uses TP=1, PP=1, BF16, an 8192-token context, audio-only multimodal limits, offline model loading, and bounded CPU threads. Its full arguments are in [`start_server.sh`](scripts/start_server.sh). Do not expose this unauthenticated example service directly to the public internet.

In another terminal, check readiness:

```bash
curl -i --max-time 30 http://127.0.0.1:8000/health
curl -fsS --max-time 30 http://127.0.0.1:8000/v1/models
```

The launcher checks port availability before loading weights. Wait for model loading and first-run GPU compilation; a transient 503 during compilation can be retried once the server settles. Use HTTP 200 from `/health` as the readiness signal.

## 5. Transcribe one audio file

Open another terminal at the repository root. The standard OpenAI-compatible endpoint is intended for ordinary client use:

```bash
export AUDIO=/absolute/path/to/speech.wav
curl --fail-with-body --max-time 600 \
  http://127.0.0.1:8000/v1/audio/transcriptions \
  -F "file=@${AUDIO}" -F 'model=ming-asr' \
  -F 'response_format=json' -F 'temperature=0'
```

The response contains a `text` field. A dependency-free client is also included. The repository's sample audio files produce `漓江和湘江有秦代开凿的灵渠沟通` and `same way you did`:

```bash
python cookbook/sglang/scripts/transcribe.py \
  --audio "$AUDIO" --base-url http://127.0.0.1:8000 \
  --model ming-asr --save-json transcription.json
```

The client can run on a different machine: only the server needs the GPU, checkpoint, and SGLang environment. The uploaded file is read on the **client** machine. For a remote server listening only on loopback, use an SSH tunnel rather than exposing an unauthenticated port.

The OpenAI Python SDK is optional. With `openai==2.6.1`, set `base_url="http://127.0.0.1:8000/v1"` and use `client.audio.transcriptions.create(model="ming-asr", file=audio, temperature=0)`. The standard-library client above requires no SDK installation.

## 6. Dataset evaluation request

The standard transcription endpoint and the dataset evaluation request use different prompts and decoding settings. The standard endpoint uses `Transcribe this speech.<AUDIO>` and a duration-dependent generation limit of at least 448. For the evaluation settings and scores below, activate the serving environment in the client terminal and run the included `/generate` client:

```bash
source .venv-sglang/bin/activate
unset PYTHONPATH
export SGLANG_DIR="$(cd ../sglang_ming_asr && pwd)" # or the absolute path to your checkout
python cookbook/sglang/scripts/transcribe_aligned.py \
  --sglang-dir "$SGLANG_DIR" \
  --audio "$AUDIO" --base-url http://127.0.0.1:8000 \
  --save-json aligned-transcription.json
```

It sends the byte-exact HF ASR instruction followed by `<AUDIO>`, one WAV data URI, temperature 0, `max_new_tokens=256`, stop token 156892, and the pinned adapter's 10-gram no-repeat processor. It removes the generated language label from the displayed transcript. Do not add a chat template, language hint, hotwords, or Context ASR prompt when reproducing the ordinary-ASR scores. The custom logit processor is a **trusted evaluation-client** feature; do not relay untrusted serialized processors to the server.

## 7. ASR quality

Results on AISHELL-1 (7,176 samples), LibriSpeech test-clean (2,620), and test-other (2,939) cover **12,735 samples total**. Both columns use the same references and normalization:

| Dataset | SGLang `/generate` | HF Transformers |
| --- | ---: | ---: |
| AISHELL-1 CER | 1.058560% | 1.044242% |
| LibriSpeech test-clean WER | 1.169216% | 1.169216% |
| LibriSpeech test-other WER | 2.405355% | 2.394009% |

These dataset results use the `/generate` request shown above. The standard transcription endpoint is intended for application use.
