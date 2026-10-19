"""Batch resume must not silently mix samples or decoding options."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))
SPEC = importlib.util.spec_from_file_location("transcribe", ROOT / "examples" / "transcribe.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_completed_prefix_validates_ids_and_options(tmp_path):
    audio = tmp_path / "speech.wav"
    rows = [("one", audio, None), ("two", audio, "context")]
    partial = tmp_path / "result.jsonl.partial.jsonl"
    first = {
        "id": "one", "audio": str(audio), "prompt": None,
        "format_version": 1, "language_hint": "zh",
        "audio_chunk_frames": 6000, "context_chunks": False,
        "max_new_tokens": 128, "attention": "sdpa", "backend": "hf",
    }
    partial.write_text(json.dumps(first) + "\n", encoding="utf-8")
    assert MODULE.completed_prefix(partial, rows, "zh", 6000, 128, "sdpa") == 1
    for language, chunk_frames, tokens in (("en", 6000, 128),
                                           ("zh", 3000, 128),
                                           ("zh", 6000, 256)):
        with pytest.raises(ValueError, match="manifest/options"):
            MODULE.completed_prefix(
                partial, rows, language, chunk_frames, tokens, "sdpa",
            )
    first["id"] = "other"
    partial.write_text(json.dumps(first) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="manifest/options"):
        MODULE.completed_prefix(partial, rows, "zh", 6000, 128, "sdpa")


def test_completed_prefix_checks_context_chunk_policy(tmp_path):
    audio = tmp_path / "speech.wav"
    rows = [("one", audio, None)]
    partial = tmp_path / "result.jsonl.partial.jsonl"
    partial.write_text(json.dumps({
        "id": "one", "audio": str(audio), "prompt": None,
        "format_version": 1, "language_hint": "zh",
        "audio_chunk_frames": 3000, "context_chunks": True,
        "max_new_tokens": 256, "attention": "sdpa", "backend": "hf",
    }) + "\n", encoding="utf-8")
    assert MODULE.completed_prefix(partial, rows, "zh", None, 256, "sdpa", True) == 1
    with pytest.raises(ValueError, match="manifest/options"):
        MODULE.completed_prefix(partial, rows, "zh", None, 256, "sdpa")
