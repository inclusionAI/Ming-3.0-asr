"""Check that a downloaded Hugging Face model directory is complete (read only)."""
import argparse
import json
from pathlib import Path


REQUIRED = (
    "asr_request.py", "config.json", "preprocessor_config.json",
    "tokenizer.json", "tokenizer_config.json", "model.safetensors.index.json",
    "configuration_ming_v3_asr.py", "modeling_ming_v3_asr.py",
    "modeling_ming_v3_asr_encoder.py", "processing_ming_v3_asr.py",
    "feature_extraction_ming_v3_asr.py",
)


def validate_checkpoint(checkpoint: Path) -> list[str]:
    errors = [f"Missing model file: {name}" for name in REQUIRED
              if not (checkpoint / name).is_file()]
    config = checkpoint / "config.json"
    if config.is_file():
        try:
            settings = json.loads(config.read_text(encoding="utf-8"))
            if settings.get("model_type") != "ming_v3_asr":
                errors.append("Expected ming_v3_asr model_type")
            auto_map = settings.get("auto_map", {})
            for entry, module in (
                ("AutoConfig", "configuration_ming_v3_asr."),
                ("AutoModelForCausalLM", "modeling_ming_v3_asr."),
                ("AutoProcessor", "processing_ming_v3_asr."),
            ):
                if not str(auto_map.get(entry, "")).startswith(module):
                    errors.append(f"Incorrect {entry} auto_map")
        except (ValueError, TypeError) as exc:
            errors.append(f"Invalid config: {exc}")
    preprocessor = checkpoint / "preprocessor_config.json"
    if preprocessor.is_file():
        try:
            settings = json.loads(preprocessor.read_text(encoding="utf-8"))
            if settings.get("processor_class") != "MingV3ASRProcessor":
                errors.append("Expected MingV3ASRProcessor")
        except (ValueError, TypeError) as exc:
            errors.append(f"Invalid preprocessor config: {exc}")
    index = checkpoint / "model.safetensors.index.json"
    if index.is_file():
        try:
            weight_map = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
            if not weight_map:
                errors.append("Empty weight_map")
            if any(name.startswith("model.visual.") for name in weight_map):
                errors.append("Unexpected vision encoder weights")
            if not any(name.startswith("model.audio.") for name in weight_map):
                errors.append("Missing audio encoder weights")
            for name in set(weight_map.values()):
                path = (checkpoint / name).resolve()
                if not path.is_relative_to(checkpoint) or not path.is_file():
                    errors.append(f"Missing or unsafe weight shard: {name}")
        except (ValueError, KeyError, TypeError) as exc:
            errors.append(f"Invalid weight index: {exc}")
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve()
    errors = validate_checkpoint(checkpoint)
    if errors:
        print("FAILED\n" + "\n".join(errors))
        return 1
    print("PASS: audio-only model files, configuration, and weight shards are present.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
