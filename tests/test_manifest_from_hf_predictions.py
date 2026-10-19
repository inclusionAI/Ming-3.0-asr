"""HF prediction export must preserve the exact evaluation sample contract."""

import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "manifest_from_hf_predictions.py"
SPEC = importlib.util.spec_from_file_location("manifest_from_hf_predictions", SCRIPT)
MANIFEST = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MANIFEST)


def write_part(directory, name, rows):
    directory.mkdir(exist_ok=True)
    (directory / name).write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def test_manifest_restores_index_order_and_omits_predictions(tmp_path):
    audio = tmp_path / "audio.wav"
    audio.touch()
    write_part(tmp_path, "part_00.jsonl", [
        {"index": 1, "id": "b", "audio": str(audio), "reference": "", "prediction": "wrong"},
    ])
    write_part(tmp_path, "part_01.jsonl", [
        {"index": 0, "id": "a", "audio": str(audio), "reference": "hello", "prediction": "hi"},
    ])
    assert MANIFEST.build_rows(tmp_path) == [
        {"id": "a", "audio": str(audio), "reference": "hello"},
        {"id": "b", "audio": str(audio), "reference": ""},
    ]


@pytest.mark.parametrize("indices,pattern", [([0, 0], "duplicate"), ([0, 2], "contiguous")])
def test_manifest_rejects_duplicate_or_missing_index(tmp_path, indices, pattern):
    audio = tmp_path / "audio.wav"
    audio.touch()
    write_part(tmp_path, "part_00.jsonl", [
        {"index": index, "id": str(pos), "audio": str(audio), "reference": "test"}
        for pos, index in enumerate(indices)
    ])
    with pytest.raises(ValueError, match=pattern):
        MANIFEST.build_rows(tmp_path)


def test_manifest_rejects_missing_audio(tmp_path):
    write_part(tmp_path, "part_00.jsonl", [
        {"index": 0, "id": "a", "audio": str(tmp_path / "missing.wav"), "reference": "test"},
    ])
    with pytest.raises(FileNotFoundError):
        MANIFEST.build_rows(tmp_path)


def test_manifest_rejects_empty_parts(tmp_path):
    write_part(tmp_path, "part_00.jsonl", [])
    with pytest.raises(ValueError, match="no rows"):
        MANIFEST.build_rows(tmp_path)
