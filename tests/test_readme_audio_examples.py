"""README media references must map to the committed sample files."""

import json
import re
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


MEDIA = {
    "intro": "366d5c76-7d7a-4c74-9315-d3b1ca1923a0",
    "mandarin": "a13297d3-1458-406d-8846-585d0cbf37b8",
    "english": "620afa11-e8d9-4550-ae64-809c36dfcd8b",
    "context": "62613e2e-6dfa-4f1d-aaa1-ab83dcd78aed",
}


@pytest.mark.parametrize("readme, expected", (
    (ROOT / "README.md", 4),
    (ROOT / "examples/cases/README.md", 3),
))
def test_inline_media_and_sample_files(readme, expected):
    contents = readme.read_text(encoding="utf-8")
    urls = re.findall(r"^https://github.com/user-attachments/assets/([0-9a-f-]{36})$", contents, re.M)
    assert len(urls) == expected
    expected_urls = set(MEDIA.values())
    if expected == 3:
        expected_urls.remove(MEDIA["intro"])
    assert set(urls) == expected_urls
    for name in ("chinese_common_voice.wav", "english_common_voice.wav", "context_same_sentence_pcm16.wav"):
        assert (ROOT / "examples/cases/audio" / name).is_file()


def test_demo_follows_model_introduction():
    contents = (ROOT / "README.md").read_text(encoding="utf-8")
    assert (
        contents.index("## Quick Start")
        < contents.index("### 5. SGLang Accelerated Serving")
        < contents.index('<a id="cookbook"></a>')
        < contents.index(f"https://github.com/user-attachments/assets/{MEDIA['mandarin']}")
    )
    assert "git clone https://github.com/inclusionAI/Ming-3.0-asr.git" in contents
    assert "Real speech comes with accents" in contents


def test_english_video_is_inline_and_not_committed():
    contents = (ROOT / "README.md").read_text(encoding="utf-8")
    assert f"https://github.com/user-attachments/assets/{MEDIA['intro']}" in contents
    assert not list(ROOT.glob("*.mp4"))


def test_context_demo_uses_same_audio_for_both_requests():
    manifest = ROOT / "examples/cases/context.jsonl"
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2
    assert rows[0]["audio"] == rows[1]["audio"]
    assert (manifest.parent / rows[0]["audio"]).is_file()
    assert rows[0]["prompt"] is None
    assert "同囚" in rows[1]["prompt"]


@pytest.mark.parametrize("readme, topic_label, hotword_label", (
    (ROOT / "README.md", "Topic", "Hotwords"),
    (ROOT / "examples/cases/README.md", "主题 / Topic", "热词 / Hotwords"),
))
def test_context_demo_displays_actual_hotwords_and_stacked_results(readme, topic_label, hotword_label):
    prompt = json.loads((ROOT / "examples/cases/context.jsonl").read_text(encoding="utf-8").splitlines()[1])["prompt"]
    hotwords = prompt.split("phrases:", 1)[1].rstrip(".").split(",")
    contents = readme.read_text(encoding="utf-8")
    assert all(word in contents for word in hotwords)
    assert "| 你们看了黄国权那部**同囚**没我昨晚看完到现在还缓不过来 |" in contents
    assert "| 你们看了黄国权那部**铜球**没我昨晚看完到现在还缓不过来 |" in contents
    reference = contents.index("你们看了黄国权那部**同囚**没我昨晚看完到现在还缓不过来 |")
    topic = contents.index(f"| {topic_label} | 同囚 |", reference)
    hotword = contents.index(f"| {hotword_label} |", topic)
    output = contents.index("你们看了黄国权那部**铜球**没我昨晚看完到现在还缓不过来 |", hotword)
    assert reference < topic < hotword < output
