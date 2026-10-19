# Evaluation and Reproduction

The [benchmark figures](../README.md#evaluation) show the released results. The tools below use their own scoring rules; comparisons require the same datasets, decoding settings, and text normalization.

## Prepare a Manifest

Create `evaluation.jsonl` with one example per line:

```json
{"id":"zh-001","audio":"audio/zh-001.wav","reference":"今天天气很好"}
{"id":"en-001","audio":"audio/en-001.wav","reference":"Good morning"}
{"id":"silence-001","audio":"audio/silence-001.wav","reference":""}
```

Audio paths are relative to the manifest file. References should come from human annotation or an official dataset. Use an empty string for silence.

## Batch Transcription with PyTorch

To transcribe multiple files locally with the HF model, create a JSONL manifest containing `id` and `audio` for each item, with paths relative to the manifest, then run:

```bash
python examples/transcribe.py \
  --model "$MODEL_DIR" \
  --manifest /path/to/transcription.jsonl \
  --output outputs/predictions.jsonl \
  --max-new-tokens 256
```

The default request uses the released ASR prompt format, BF16, SDPA, greedy decoding, a 10-gram repetition limit, and `max_new_tokens=256`. Ordinary ASR does not add a language hint, reference, or hotwords. For Context ASR, supply the full prompt beginning with the base instruction in the manifest's `prompt` field; the audio format remains the same. Output includes decoding settings. This script transcribes only; computing dataset-level WER/CER also requires the dataset's split, normalization, and scoring rules.

## Evaluate a Serving Endpoint

The evaluation tool expects compatible `/v1/models` and `/v1/audio/transcriptions` endpoints. `--model auto` queries the served model. Install the optional dependencies first:

```bash
python -m pip install -r requirements-dev.txt
python tools/evaluate_asr.py \
  --manifest /path/to/evaluation.jsonl \
  --base-url http://127.0.0.1:8000 \
  --model auto --output outputs/evaluation.json
```

This tool does not start the server and has no authentication option. It is for compatible transcription servers that do not require authentication; a `/v1/completions`-only endpoint is not sufficient.

## Interpret the Metrics

- Normalization applies NFKC, casefolding, Unicode punctuation removal, and whitespace collapse. CER also removes spaces.
- WER tokenizes on whitespace. For continuous Chinese text, use CER rather than treating an entire sentence as one WER token.
- WER/CER outputs are ratios, not percentages. Multiply by 100 for percentages; insertions can make an error ratio exceed 1.
- Corpus scores divide total edit distance by total reference length, rather than averaging per-sample error rates.
- Empty references still count insertion errors. An item with a zero denominator receives `null`.
- Failed requests mark the report `incomplete` and produce a nonzero exit code; aggregate scores cover successful requests only.
- RTF is total elapsed time for serial requests divided by total audio duration, not kernel time or concurrent throughput.
- This tool does not compute NE-WER/NE-FNR. Context ASR requires entity annotations and a separate evaluation protocol.

[Back to Cookbook](README.md)
