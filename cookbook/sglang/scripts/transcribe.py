"""Transcribe one WAV through the OpenAI-compatible ASR endpoint; stdlib only."""
import argparse
import json
import mimetypes
from pathlib import Path
import sys
import urllib.error
import urllib.request
import uuid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="ming-asr")
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--save-json", type=Path)
    args = parser.parse_args()
    if not args.audio.is_file():
        parser.error(f"Audio file does not exist on the client: {args.audio}")
    boundary = "ming-" + uuid.uuid4().hex
    body = bytearray()
    for key, value in {"model": args.model, "response_format": "json", "temperature": "0"}.items():
        body.extend(f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"\r\n\r\n{value}\r\n'.encode())
    filename = args.audio.name.replace('"', "_").replace("\r", "_").replace("\n", "_")
    mime = mimetypes.guess_type(filename)[0] or "application/octet-stream"
    body.extend(f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{filename}"\r\nContent-Type: {mime}\r\n\r\n'.encode())
    body.extend(args.audio.read_bytes())
    body.extend(f"\r\n--{boundary}--\r\n".encode())
    request = urllib.request.Request(args.base_url.rstrip("/") + "/v1/audio/transcriptions", data=bytes(body), headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    try:
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            result = json.load(response)
    except urllib.error.HTTPError as error:
        print(f"HTTP {error.code}: {error.read().decode(errors='replace')}", file=sys.stderr)
        return 1
    except (urllib.error.URLError, TimeoutError) as error:
        print(f"Request failed: {error}", file=sys.stderr)
        return 1
    if not isinstance(result.get("text"), str) or not result["text"].strip():
        print(f"No nonempty transcript in response: {result}", file=sys.stderr)
        return 1
    print(result["text"])
    if args.save_json:
        args.save_json.parent.mkdir(parents=True, exist_ok=True)
        args.save_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
