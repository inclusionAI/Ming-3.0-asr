"""Keep the public repository's core release metadata consistent."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_license_and_citation_match():
    assert (ROOT / "LICENSE").read_text(encoding="utf-8").startswith("MIT License\n")
    citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    assert "license: MIT\n" in citation
    assert "name: inclusionAI\n" in citation


def test_main_readme_uses_english_and_15_dialect_sets():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "evaluated on 15 test sets" in readme
    assert "Yunnan" in readme
    assert "## Quick Start" in readme
    assert "## Evaluation" in readme
    assert not (ROOT / "README.en.md").exists()
