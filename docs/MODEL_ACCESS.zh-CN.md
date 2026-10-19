# 获取模型

完整模型包位于 [Hugging Face：inclusionAI/Ming-3.0-asr-tiny](https://huggingface.co/inclusionAI/Ming-3.0-asr-tiny)。本代码仓库不镜像模型源码、配置、tokenizer 或权重；请从 HF 模型仓库下载完整模型包。

```bash
python -m pip install "huggingface_hub[cli]"
hf download inclusionAI/Ming-3.0-asr-tiny --local-dir ./Ming-3.0-asr-tiny-model
```

下载后，可按 [PyTorch / Transformers 推理示例](../cookbook/hf-inference.md)运行。不要仅复制部分权重分片或将训练检查点直接传给推理脚本。

[返回首页](../README.md)
