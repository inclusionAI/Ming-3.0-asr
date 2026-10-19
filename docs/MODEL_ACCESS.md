# Get the Model

The complete model package is available at [Hugging Face: inclusionAI/Ming-3.0-asr-tiny](https://huggingface.co/inclusionAI/Ming-3.0-asr-tiny). This code repository does not mirror the model implementation, configuration, tokenizer, or weights. Download the complete package from Hugging Face:

```bash
python -m pip install "huggingface_hub[cli]"
hf download inclusionAI/Ming-3.0-asr-tiny --local-dir ./Ming-3.0-asr-tiny-model
```

Then follow the [PyTorch / Transformers inference guide](../cookbook/hf-inference.md). Do not copy only some weight shards or pass a training checkpoint directly to the inference script.

[Back to Model README](../README.md)
