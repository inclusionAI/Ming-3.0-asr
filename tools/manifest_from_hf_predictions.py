"""Build an ASR evaluation manifest from existing HF part_*.jsonl outputs.

The resulting manifest reuses the same audio, reference, ID and original row order;
predictions are deliberately omitted so the manifest is backend-neutral.
"""

import argparse
import json
from pathlib import Path


def build_rows(directory: Path, check_audio: bool = True) -> list[dict]:
    parts = sorted(directory.glob("part_*.jsonl"))
    if not parts:
        raise ValueError(f"No part_*.jsonl files in {directory}")
    indexed = {}
    ids = set()
    for part in parts:
        for line_no, line in enumerate(part.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            record = json.loads(line)
            index = record.get("index")
            uid = record.get("id")
            audio = record.get("audio")
            reference = record.get("reference")
            if not isinstance(index, int) or index < 0 or isinstance(index, bool):
                raise ValueError(f"{part}:{line_no}: missing nonnegative integer index")
            if not (isinstance(uid, str) and uid and isinstance(audio, str) and audio
                    and isinstance(reference, str)):
                raise ValueError(f"{part}:{line_no}: missing id, audio or reference")
            if index in indexed or uid in ids:
                raise ValueError(f"{part}:{line_no}: duplicate index or ID")
            if check_audio and not Path(audio).is_file():
                raise FileNotFoundError(audio)
            indexed[index] = {"id": uid, "audio": audio, "reference": reference}
            ids.add(uid)
    if not indexed:
        raise ValueError("HF prediction parts contain no rows")
    if sorted(indexed) != list(range(len(indexed))):
        raise ValueError("HF rows must have contiguous zero-based indices")
    return [indexed[index] for index in range(len(indexed))]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skip-audio-check", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    rows = build_rows(args.hf_dir, check_audio=not args.skip_audio_check)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
