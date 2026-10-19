"""Evaluate an OpenAI-compatible ASR endpoint against an explicit JSONL manifest."""

import argparse
import hashlib
import json
import time
import unicodedata
from pathlib import Path

NORMALIZATION = "nfkc-casefold-remove-unicode-punctuation-collapse-whitespace-v1"


def normalize(text):
    text = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(
        "".join(c for c in text if not unicodedata.category(c).startswith("P")).split()
    )


def edit_distance(reference, hypothesis):
    previous = list(range(len(hypothesis) + 1))
    for i, ref in enumerate(reference, 1):
        current = [i]
        for j, hyp in enumerate(hypothesis, 1):
            current.append(
                min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (ref != hyp))
            )
        previous = current
    return previous[-1]


def score(reference, hypothesis):
    ref, hyp = normalize(reference), normalize(hypothesis)
    words, chars = ref.split(), ref.replace(" ", "")
    we = edit_distance(words, hyp.split())
    ce = edit_distance(chars, hyp.replace(" ", ""))
    return {
        "word_edits": we,
        "reference_words": len(words),
        "char_edits": ce,
        "reference_chars": len(chars),
        "wer": we / len(words) if words else None,
        "cer": ce / len(chars) if chars else None,
    }


def summarize(rows, *, expected_count=None):
    expected_count = len(rows) if expected_count is None else expected_count
    if expected_count == 0:
        raise ValueError("Evaluation manifest is empty")
    if expected_count < len(rows):
        raise ValueError("More results than expected manifest rows")
    successful = [r for r in rows if r["status"] == "passed"]
    sums = {
        k: sum(r[k] for r in successful)
        for k in (
            "word_edits",
            "reference_words",
            "char_edits",
            "reference_chars",
            "duration_s",
            "elapsed_s",
        )
    }
    failed = len(rows) - len(successful)
    pending = expected_count - len(rows)
    return {
        "status": "in_progress" if pending else "incomplete" if failed else "complete",
        "expected": expected_count,
        "attempted": len(rows),
        "pending": pending,
        "completed": len(successful),
        "failed": failed,
        "metric_scope": "successful_requests_only",
        **sums,
        "wer": sums["word_edits"] / sums["reference_words"]
        if sums["reference_words"]
        else None,
        "cer": sums["char_edits"] / sums["reference_chars"]
        if sums["reference_chars"]
        else None,
        "rtf": sums["elapsed_s"] / sums["duration_s"] if sums["duration_s"] else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="JSONL: id, audio (relative to manifest), reference",
    )
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", default="auto")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=180)
    args = parser.parse_args()
    import requests
    import soundfile as sf

    records = [
        json.loads(line)
        for line in args.manifest.read_text().splitlines()
        if line.strip()
    ]
    if not records:
        parser.error("Evaluation manifest is empty")
    if not all(isinstance(r.get("reference"), str) and r.get("audio") for r in records):
        parser.error(
            "Every manifest row needs audio and a reference string (empty for silence)"
        )
    report = {
        "normalization": NORMALIZATION,
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "base_url": args.base_url,
        "model": args.model,
        "temperature": 0,
        "rows": [],
    }
    session = requests.Session()
    session.trust_env = False
    base = args.base_url.rstrip("/").removesuffix("/v1")
    report["summary"] = summarize([], expected_count=len(records))
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    for record in records:
        row = {
            "id": record.get("id"),
            "audio": record["audio"],
            "reference": record["reference"],
        }
        try:
            audio = args.manifest.parent / record["audio"]
            info = sf.info(audio)
            row["audio_sha256"] = hashlib.sha256(audio.read_bytes()).hexdigest()
            row["duration_s"] = info.frames / info.samplerate
            started = time.monotonic()
            with audio.open("rb") as handle:
                response = session.post(
                    base + "/v1/audio/transcriptions",
                    files={"file": (audio.name, handle)},
                    data={
                        "model": args.model,
                        "temperature": 0,
                        "response_format": "json",
                    },
                    timeout=(10, args.timeout),
                )
            response.raise_for_status()
            row["elapsed_s"] = time.monotonic() - started
            hypothesis = response.json()["text"]
            row.update(
                status="passed",
                hypothesis=hypothesis,
                **score(row["reference"], hypothesis),
            )
        except Exception as error:
            row.update(status="failed", error=str(error))
        report["rows"].append(row)
        report["summary"] = summarize(report["rows"], expected_count=len(records))
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        print(json.dumps({"id": row["id"], "status": row["status"]}), flush=True)
    print(json.dumps(report["summary"], indent=2))
    return bool(report["summary"]["failed"])


if __name__ == "__main__":
    raise SystemExit(main())
