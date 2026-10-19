#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_DIR:?Set MODEL_DIR to a complete local checkpoint directory}"
: "${SGLANG_DIR:?Set SGLANG_DIR to the pinned SGLang source repository}"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_ID="${GPU_ID:-0}"
PORT="${PORT:-8000}"
HOST="${HOST:-127.0.0.1}"
SERVED_MODEL="${SERVED_MODEL:-ming-asr}"
test -f "$MODEL_DIR/config.json"
test -f "$SGLANG_DIR/python/sglang/launch_server.py"
"$PYTHON_BIN" - "$HOST" "$PORT" <<'PY'
import socket
import sys

host, port = sys.argv[1], int(sys.argv[2])
with socket.socket() as listener:
    try:
        listener.bind((host, port))
    except OSError as exc:
        raise SystemExit(f"Cannot listen on {host}:{port}: {exc}. Choose another PORT.")
PY

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export PYTHONPATH="$SGLANG_DIR/python:$MODEL_DIR"
export OMP_NUM_THREADS="${CPU_THREADS:-4}"
export MKL_NUM_THREADS="${CPU_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${BLAS_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${CPU_THREADS:-4}"
export SGLANG_IO_WORKERS="${SGLANG_IO_WORKERS:-8}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

exec "$PYTHON_BIN" -u -m sglang.launch_server \
  --model-path "$MODEL_DIR" \
  --served-model-name "$SERVED_MODEL" \
  --host "$HOST" --port "$PORT" \
  --tp-size 1 --pp-size 1 --nnodes 1 \
  --context-length 8192 --dtype bfloat16 \
  --mem-fraction-static "${MEM_FRACTION:-0.80}" \
  --max-running-requests "${MAX_RUNNING_REQUESTS:-64}" \
  --max-prefill-tokens 16384 --chunked-prefill-size 2048 \
  --tokenizer-worker-num 1 \
  --enable-multimodal \
  --limit-mm-data-per-request '{"image":0,"video":0,"audio":1}' \
  --trust-remote-code --enable-custom-logit-processor \
  --skip-server-warmup "$@"
