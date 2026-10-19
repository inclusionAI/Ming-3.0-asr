"""Use the exact /generate prompt and decoding settings of the alignment test."""
import argparse
import base64
import json
import os
from pathlib import Path
import sys
import urllib.error
import urllib.request

ASR_INSTRUCTION = (
    "Please recognize the language of this speech and transcribe it, "
    "placing a `\t` between the identified language and the transcribe."
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--sglang-dir", default=os.environ.get("SGLANG_DIR"))
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--save-json", type=Path)
    args = parser.parse_args()
    if not args.audio.is_file():
        parser.error(f"Audio file does not exist: {args.audio}")
    if args.audio.suffix.lower() != ".wav":
        parser.error("This alignment example expects a real WAV file, not a renamed MP3")
    if args.sglang_dir:
        sys.path.insert(0, str(Path(args.sglang_dir).resolve() / "python"))
    from sglang.srt.sampling.custom_logit_processor import DeepseekOCRNoRepeatNGramLogitProcessor

    payload = {
        "text": ASR_INSTRUCTION + "<AUDIO>",
        "audio_data": ["data:audio/wav;base64," + base64.b64encode(args.audio.read_bytes()).decode()],
        "sampling_params": {
            "temperature": 0, "max_new_tokens": 256,
            "stop_token_ids": [156892], "skip_special_tokens": True,
            "custom_params": {"ngram_size": 10, "window_size": 8192},
        },
        "custom_logit_processor": DeepseekOCRNoRepeatNGramLogitProcessor.to_str(),
        "return_logprob": True, "logprob_start_len": -1, "top_logprobs_num": 0,
        "return_prompt_token_ids": True,
    }
    request = urllib.request.Request(args.base_url.rstrip("/") + "/generate", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            result = json.load(response)
    except urllib.error.HTTPError as error:
        print(f"HTTP {error.code}: {error.read().decode(errors='replace')}", file=sys.stderr)
        return 1
    except (urllib.error.URLError, TimeoutError) as error:
        print(f"Request failed: {error}", file=sys.stderr)
        return 1
    text = result.get("text", "").strip()
    if "\t" in text:
        text = text.split("\t", 1)[1].strip()
    else:
        for language in ("Chinese", "English", "Canton", "Japanese", "Korean"):
            if text.startswith(language):
                text = text[len(language):].strip()
                break
    if not text:
        print(f"Empty transcript: {result}", file=sys.stderr)
        return 1
    print(text)
    if args.save_json:
        args.save_json.parent.mkdir(parents=True, exist_ok=True)
        args.save_json.write_text(json.dumps({"transcript": text, "raw_response": result}, ensure_ascii=False, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
