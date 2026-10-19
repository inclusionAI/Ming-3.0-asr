"""Transcribe one audio file or a JSONL manifest with the HF model."""

import argparse
import json
from pathlib import Path
from time import perf_counter

from hf_runtime import (
    context_chunk_frames, decode_transcript, prepare_asr_request,
)


def read_rows(path: Path) -> list[tuple[str, Path, str | None]]:
    """Read a manifest of unique IDs, audio paths, and optional prompts."""
    rows = []
    seen = set()
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        uid, audio, prompt = row.get("id"), row.get("audio"), row.get("prompt")
        if not isinstance(uid, str) or not uid or uid in seen:
            raise ValueError(f"{path}:{line_no}: ID must be a unique nonempty string")
        if not isinstance(audio, str) or not audio:
            raise ValueError(f"{path}:{line_no}: audio must be a nonempty path")
        if prompt is not None and not isinstance(prompt, str):
            raise ValueError(f"{path}:{line_no}: prompt must be a string or null")
        audio_path = Path(audio)
        if not audio_path.is_absolute():
            audio_path = path.parent / audio_path
        if not audio_path.is_file():
            raise FileNotFoundError(audio_path)
        rows.append((uid, audio_path, prompt))
        seen.add(uid)
    if not rows:
        raise ValueError(f"{path}: no input rows")
    return rows


def completed_prefix(partial, rows, language_hint, audio_chunk_frames,
                     max_new_tokens, attention, context_chunks=False):
    """Only resume a partial file that matches this ordered manifest and protocol."""
    existing = [json.loads(line) for line in partial.read_text(encoding="utf-8").splitlines()]
    if len(existing) > len(rows):
        raise ValueError("Partial output has more rows than the manifest")
    for index, row in enumerate(existing):
        uid, audio, prompt = rows[index]
        if (row.get("id") != uid or row.get("audio") != str(audio)
                or row.get("prompt") != prompt
                or row.get("format_version") != 1
                or row.get("language_hint") != language_hint
                or row.get("context_chunks", False) != context_chunks
                or (not context_chunks
                    and row.get("audio_chunk_frames") != audio_chunk_frames)
                or row.get("max_new_tokens") != max_new_tokens
                or row.get("attention") != attention
                or row.get("backend") != "hf"):
            raise ValueError(f"Partial output does not match manifest/options at row {index + 1}")
    return len(existing)


def transcribe_one(processor, model, torch, sf, audio, prompt, args,
                   max_new_tokens, default_chunk_frames):
    waveform, rate = sf.read(audio, dtype="float32", always_2d=True)
    duration = len(waveform) / rate
    chunk_frames = (context_chunk_frames(duration)
                    if args.context_chunks else args.audio_chunk_frames)
    model.config.audio_config.chunk_frames = (
        chunk_frames if chunk_frames is not None else default_chunk_frames
    )
    inputs = prepare_asr_request(
        processor, (waveform.T, rate), prompt=prompt,
        language_hint=args.language_hint, audio_chunk_frames=chunk_frames,
    ).to(model.device)
    eos = getattr(model.config, "text_config", model.config).eos_token_id
    with torch.inference_mode():
        output = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            eos_token_id=eos, pad_token_id=eos, no_repeat_ngram_size=10,
        )
    tokens = output[0, inputs.input_ids.shape[1]:].tolist()
    text = decode_transcript(processor.tokenizer, tokens)
    return {
        "audio": str(audio), "text": text, "prompt": prompt,
        "format_version": 1,
        "language_hint": args.language_hint,
        "audio_chunk_frames": chunk_frames,
        "context_chunks": args.context_chunks,
        "max_new_tokens": max_new_tokens, "attention": args.attention,
        "generated_token_ids": tokens, "backend": "hf",
    }, duration


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True,
                        help="Downloaded HF model directory")
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--audio", type=Path, help="One audio file")
    inputs.add_argument("--manifest", type=Path,
                        help="JSONL rows with id, audio, and optional prompt")
    parser.add_argument("--output", type=Path,
                        help="Optional JSON for --audio; required JSONL for --manifest")
    parser.add_argument("--resume", action="store_true",
                        help="Resume a matching *.partial.jsonl for --manifest")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--attention", choices=("sdpa", "eager"), default="sdpa")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--prompt", help="Full context prompt for --audio")
    parser.add_argument("--language-hint", choices=("zh", "en"),
                        help="Known-language prefix; omit when language is unknown")
    parser.add_argument("--audio-chunk-frames", type=int,
                        help="Fixed mel frames per encoder chunk")
    parser.add_argument("--context-chunks", action="store_true",
                        help="Context ASR: 3000 frames up to 30 s, else 6000")
    args = parser.parse_args()

    if not args.model.is_dir():
        parser.error(f"HF model directory does not exist: {args.model}")
    if args.audio and not args.audio.is_file():
        parser.error(f"Audio file does not exist: {args.audio}")
    if args.manifest and args.prompt is not None:
        parser.error("Put per-audio prompts in the manifest; --prompt is for --audio")
    if args.manifest and args.output is None:
        parser.error("--manifest requires --output")
    if args.resume and not args.manifest:
        parser.error("--resume requires --manifest")
    if args.max_new_tokens is not None and args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive")
    if args.audio_chunk_frames is not None and args.audio_chunk_frames < 1:
        parser.error("--audio-chunk-frames must be positive")
    if args.context_chunks and args.audio_chunk_frames is not None:
        parser.error("Use either --context-chunks or --audio-chunk-frames")
    max_new_tokens = args.max_new_tokens

    rows = read_rows(args.manifest) if args.manifest else [(None, args.audio, args.prompt)]
    partial = (args.output.with_suffix(args.output.suffix + ".partial.jsonl")
               if args.manifest else None)
    completed = 0
    if partial:
        if args.output.exists() or (partial.exists() and not args.resume):
            parser.error("Output already exists; choose a new --output path or --resume a partial")
        if partial.exists():
            completed = completed_prefix(
                partial, rows, args.language_hint,
                args.audio_chunk_frames, max_new_tokens, args.attention,
                args.context_chunks,
            )

    import soundfile as sf
    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor

    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        parser.error("The supported inference path requires a CUDA GPU")
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, dtype=torch.bfloat16,
        device_map=args.device, attn_implementation=args.attention,
    ).eval()
    default_chunk_frames = model.config.audio_config.chunk_frames
    started = perf_counter()
    audio_seconds = 0.0
    if args.audio:
        result, _ = transcribe_one(
            processor, model, torch, sf, args.audio, args.prompt,
            args, max_new_tokens, default_chunk_frames,
        )
        print(result["text"])
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps({**result, "model": str(args.model)}, ensure_ascii=False, indent=2)
                + "\n", encoding="utf-8",
            )
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with partial.open("a" if partial.exists() else "x", encoding="utf-8") as stream:
        for index, (uid, audio, prompt) in enumerate(rows[completed:], completed + 1):
            result, duration = transcribe_one(
                processor, model, torch, sf, audio, prompt,
                args, max_new_tokens, default_chunk_frames,
            )
            audio_seconds += duration
            stream.write(json.dumps({"id": uid, **result}, ensure_ascii=False) + "\n")
            stream.flush()
            print(f"Transcribed {index}/{len(rows)}", flush=True)
    partial.rename(args.output)
    elapsed = perf_counter() - started
    print(json.dumps({
        "samples": len(rows), "resumed_samples": completed,
        "audio_seconds_this_run": round(audio_seconds, 3),
        "pipeline_seconds": round(elapsed, 3),
        "rtf": round(elapsed / audio_seconds, 4) if audio_seconds else None,
        "output": str(args.output),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
