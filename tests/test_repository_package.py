"""The release verifier must reject incomplete or vision-bearing packages."""

import json
import importlib.util
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "verify_repository", Path(__file__).resolve().parents[1] / "tools/verify_repository.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
REQUIRED = MODULE.REQUIRED
validate_checkpoint = MODULE.validate_checkpoint


def make_package(root: Path) -> None:
    for name in REQUIRED:
        (root / name).write_text("", encoding="utf-8")
    (root / "config.json").write_text(json.dumps({
        "model_type": "ming_v3_asr",
        "auto_map": {
            "AutoConfig": "configuration_ming_v3_asr.MingV3ASRConfig",
            "AutoModelForCausalLM": "modeling_ming_v3_asr.MingV3ASRForConditionalGeneration",
            "AutoProcessor": "processing_ming_v3_asr.MingV3ASRProcessor",
        },
    }), encoding="utf-8")
    (root / "preprocessor_config.json").write_text(
        json.dumps({"processor_class": "MingV3ASRProcessor"}), encoding="utf-8",
    )
    (root / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {"model.audio.weight": "model.safetensors"},
    }), encoding="utf-8")
    (root / "model.safetensors").write_bytes(b"placeholder")


def test_accepts_audio_only_package(tmp_path):
    make_package(tmp_path)
    assert validate_checkpoint(tmp_path) == []


def test_rejects_vision_weights_and_legacy_model_type(tmp_path):
    make_package(tmp_path)
    config = json.loads((tmp_path / "config.json").read_text())
    config["model_type"] = "bailing_moe_v3_vl"
    (tmp_path / "config.json").write_text(json.dumps(config))
    index = json.loads((tmp_path / "model.safetensors.index.json").read_text())
    index["weight_map"]["model.visual.weight"] = "model.safetensors"
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))
    errors = validate_checkpoint(tmp_path)
    assert "Expected ming_v3_asr model_type" in errors
    assert "Unexpected vision encoder weights" in errors
