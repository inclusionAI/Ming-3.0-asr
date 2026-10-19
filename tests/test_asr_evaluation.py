"""Metric accounting includes silent references and never hides failed requests."""

import importlib.util
from pathlib import Path

import pytest


def evaluator():
    path = Path(__file__).resolve().parents[1] / "tools/evaluate_asr.py"
    assert path.is_file(), "Missing reproducible ASR evaluator"
    spec = importlib.util.spec_from_file_location("asr_eval", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_normalization_and_known_edit_counts():
    module = evaluator()
    result = module.score("HELLO, world!", "hello brave world")
    assert result["word_edits"] == 1 and result["reference_words"] == 2
    assert result["wer"] == 0.5
    assert module.score("你好，世界！", "你好世界")["cer"] == 0


def test_silent_references_record_hallucinations_without_division_by_zero():
    module = evaluator()
    result = module.score("", "invented words")
    assert result["reference_words"] == 0 and result["word_edits"] == 2
    assert result["wer"] is None
    assert module.score("", "")["char_edits"] == 0


def test_aggregate_uses_corpus_denominator_and_reports_failed_cases():
    module = evaluator()
    rows = [
        dict(status="passed", duration_s=2, elapsed_s=1, **module.score("a b", "a")),
        dict(status="passed", duration_s=1, elapsed_s=0.5, **module.score("", "extra")),
        dict(status="failed", error="timeout"),
    ]
    result = module.summarize(rows)
    assert result["wer"] == 1 and result["failed"] == 1
    assert result["status"] == "incomplete"
    assert result["rtf"] == 0.5


def test_empty_manifest_is_not_a_success():
    module = evaluator()
    with pytest.raises(ValueError, match="empty"):
        module.summarize([])


def test_interrupted_prefix_cannot_claim_complete_corpus():
    module = evaluator()
    row = dict(status="passed", duration_s=2, elapsed_s=1, **module.score("a", "a"))
    result = module.summarize([row], expected_count=3)
    assert result["status"] == "in_progress"
    assert (
        result["expected"] == 3 and result["pending"] == 2 and result["attempted"] == 1
    )
    initial = module.summarize([], expected_count=3)
    assert initial["pending"] == 3 and initial["wer"] is None
