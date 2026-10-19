# 评测与复现

[首页评测图](../README.md#评测结果)展示发布结果。本工具采用下述评分规则；比较不同实验时，应统一测试集、解码设置和归一化方式。

## 准备 manifest

创建 `evaluation.jsonl`，每行一个样例：

```json
{"id":"zh-001","audio":"audio/zh-001.wav","reference":"今天天气很好"}
{"id":"en-001","audio":"audio/en-001.wav","reference":"Good morning"}
{"id":"silence-001","audio":"audio/silence-001.wav","reference":""}
```

音频路径相对 manifest 所在目录。参考文本应来自人工标注或正式测试集，静音使用空字符串。

## 本地 PyTorch 批量转写

若要用 HF 模型在本地转写多条音频，可创建仅含 `id` 和 `audio` 的 JSONL 清单（路径相对清单所在目录），再运行：

```bash
python examples/transcribe.py \
  --model "$MODEL_DIR" \
  --manifest /path/to/transcription.jsonl \
  --output outputs/predictions.jsonl \
  --max-new-tokens 256
```

默认请求使用发布评测的音频提示格式、BF16、SDPA、greedy decoding、10-gram 重复限制和 `max_new_tokens=256`。普通 ASR 不添加语言提示、参考文本或热词；Context ASR 在清单的 `prompt` 字段提供以基础指令开头的完整提示，仍使用同一输入格式。输出包含解码参数。此脚本只转写，不计算 WER/CER；计算数据集指标还需使用对应测试集的切分及文本归一化、评分规则。

## 服务接口评测

本工具要求服务提供兼容的 `/v1/models` 和 `/v1/audio/transcriptions` 接口（`--model auto` 会查询模型）。先安装可选依赖：

```bash
python -m pip install -r requirements-dev.txt
python tools/evaluate_asr.py \
  --manifest /path/to/evaluation.jsonl \
  --base-url http://127.0.0.1:8000 \
  --model auto --output outputs/evaluation.json
```

工具不启动服务，也没有认证参数。它适用于无需认证的兼容转写服务；仅提供 `/v1/completions` 的接口不适用。

## 解释指标

- 归一化：NFKC、casefold、移除 Unicode 标点、压缩空白；CER 另移除空格。
- WER 按空白分词；中文连续文本通常应查看 CER，不把整句视为单词后的 WER 当作中文 CER。
- 输出 WER/CER 是 0–1 比率尺度，展示百分比时乘以 100；含插入错误时可能大于 1。
- corpus 指标按总编辑数 / 总参考长度计算，不是样本错误率的简单平均。
- 空参考保留插入错误；分母为零时单项指标为 null。
- 失败请求使报告为 incomplete 并返回非零退出码；聚合指标只覆盖成功请求。
- RTF 是串行请求的总耗时 / 音频总时长，不等于模型内核耗时或并发吞吐。
- 工具不计算 NE-WER/NE-FNR；Context ASR 需实体标注和单独评测协议。

[返回 cookbook](README.md)
