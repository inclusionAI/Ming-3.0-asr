"""Check the selected local source, checkpoint and GPU without loading weights."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=os.environ.get("MODEL_DIR"), required=not os.environ.get("MODEL_DIR"))
    parser.add_argument("--sglang-dir", default=os.environ.get("SGLANG_DIR"), required=not os.environ.get("SGLANG_DIR"))
    args = parser.parse_args()
    model = Path(args.model).expanduser().resolve()
    source = Path(args.sglang_dir).expanduser().resolve()
    if not (source / "python/sglang/launch_server.py").is_file():
        parser.error("SGLANG_DIR must contain python/sglang/launch_server.py")
    for filename in ["config.json", "tokenizer_config.json", "tokenizer.json", "preprocessor_config.json", "model.safetensors.index.json"]:
        if not (model / filename).is_file():
            parser.error(f"Incomplete model directory: missing {filename}")
    index = json.loads((model / "model.safetensors.index.json").read_text())
    for shard in set(index["weight_map"].values()):
        if not (model / shard).is_file() or (model / shard).stat().st_size == 0:
            parser.error(f"Missing or empty weight shard: {shard}")
    sys.path.insert(0, str(source / "python"))
    sys.path.insert(1, str(model))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    import torch
    import torchaudio
    import sglang
    from sglang.srt.utils.hf_transformers import get_config

    if not Path(sglang.__file__).resolve().is_relative_to(source / "python"):
        raise RuntimeError(f"Wrong SGLang source imported: {sglang.__file__}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; check the GPU container/runtime and Torch build")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("SGLang serving requires BF16 support")
    config = get_config(str(model), trust_remote_code=True)
    expected = "BailingMoeV3AudioForConditionalGeneration"
    if config.architectures != [expected]:
        raise RuntimeError(f"Config resolved to {config.architectures}, expected {[expected]}. Apply the provided compatibility patch for MingV3ASR packages.")
    result = {
        "python": sys.version.split()[0], "python_executable": sys.executable,
        "torch": torch.__version__, "torchaudio": torchaudio.__version__,
        "cuda_runtime": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
        "bf16_supported": torch.cuda.is_bf16_supported(),
        "sglang_import": sglang.__file__, "checkpoint": str(model),
        "config_class": type(config).__name__, "architectures": config.architectures,
        "versions": {name: importlib.metadata.version(name) for name in ["transformers", "sglang-kernel", "flashinfer-python", "dill"]},
        "result": "Ready to start the SGLang server",
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
