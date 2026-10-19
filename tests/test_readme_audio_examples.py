"""README media references must map to the committed sample files."""

import json
import re
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


INTRO_VIDEO = "366d5c76-7d7a-4c74-9315-d3b1ca1923a0"
AUDIO_PLAYERS = {
    "9f757b2d-c1df-4223-afb7-079b7b87bfc5",
    "6a519004-d0b3-43d9-a0d3-05caa6ed20cc",
    "eef64725-bbd2-4961-9e1c-0a65a8bede33",
}
SAMPLE_FILES = {
    "chinese_common_voice.wav",
    "english_common_voice.wav",
    "context_same_sentence_pcm16.wav",
}


@pytest.mark.parametrize("readme", (
    ROOT / "README.md",
    ROOT / "examples/cases/README.md",
))
def test_audio_links_point_to_original_wav_files(readme):
    contents = readme.read_text(encoding="utf-8")
    players = set(re.findall(r"^https://github.com/user-attachments/assets/([0-9a-f-]{36})$", contents, re.M))
    assert AUDIO_PLAYERS <= players
    paths = re.findall(r"\]\(([^)]+\.wav\?raw=1)\)", contents)
    assert len(paths) == 3
    assert {Path(path.removesuffix("?raw=1")).name for path in paths} == SAMPLE_FILES
    for path in paths:
        assert (readme.parent / path.removesuffix("?raw=1")).is_file()
    assert "<audio" not in contents
    if readme != ROOT / "README.md":
        assert players == AUDIO_PLAYERS


def test_demo_follows_model_introduction():
    contents = (ROOT / "README.md").read_text(encoding="utf-8")
    assert (
        contents.index("## Quick Start")
        < contents.index("### 5. SGLang Accelerated Serving")
        < contents.index('<a id="cookbook"></a>')
        < contents.index("https://github.com/user-attachments/assets/9f757b2d-c1df-4223-afb7-079b7b87bfc5")
    )
    assert "git clone https://github.com/inclusionAI/Ming-3.0-asr.git" in contents
    assert "Real speech comes with accents" in contents


def test_english_video_is_inline_and_not_committed():
    contents = (ROOT / "README.md").read_text(encoding="utf-8")
    assert f"https://github.com/user-attachments/assets/{INTRO_VIDEO}" in contents
    assert contents.count("https://github.com/user-attachments/assets/") == 4
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
