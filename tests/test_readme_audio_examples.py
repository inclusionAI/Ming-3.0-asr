"""The embedded audio players must point to committed sample files."""

import json
import re
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("readme, expected", (
    (ROOT / "README.md", 3),
    (ROOT / "examples/cases/README.md", 3),
))
def test_embedded_audio_players_resolve(readme, expected):
    paths = re.findall(
        r'<audio\b[^>]*\bcontrols\b[^>]*\bsrc="([^"]+\.wav)"',
        readme.read_text(encoding="utf-8"),
    )
    assert len(paths) == expected
    for path in paths:
        if readme == ROOT / "README.md":
            prefix = "https://raw.githubusercontent.com/inclusionAI/Ming-3.0-asr/main/"
        else:
            prefix = "/zhuangxuyi.zxy/Ming-3.0-asr-tiny/-/raw/master/"
        assert path.startswith(prefix)
        assert (ROOT / path.removeprefix(prefix)).is_file()


def test_demo_follows_model_introduction():
    contents = (ROOT / "README.md").read_text(encoding="utf-8")
    assert (
        contents.index("## Quick Start")
        < contents.index("### 5. SGLang Accelerated Serving")
        < contents.index('<a id="cookbook"></a>')
        < contents.index("<audio controls")
    )
    assert "git clone https://github.com/inclusionAI/Ming-3.0-asr.git" in contents
    assert "Real speech comes with accents" in contents


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
