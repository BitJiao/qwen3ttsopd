#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
export PYTHONPATH="${ROOT}:${QWEN3_TTS_REPO:-${ROOT}/../Qwen3-TTS-main}:${PYTHONPATH:-}"

"${PYTHON:-${ROOT}/.venv/bin/python}" -m qwen3opsd.eval_emotiontalk \
  --model-path "${MODEL_PATH:?set MODEL_PATH}" \
  --model-name "${MODEL_NAME:?set MODEL_NAME}" \
  --input-jsonl "${INPUT_JSONL:-data/processed/emotiontalk/sft_test.jsonl}" \
  --output-dir "${OUTPUT_DIR:?set OUTPUT_DIR}" \
  --conditioning "${CONDITIONING:-instruction}" \
  --device "${DEVICE:-cuda:0}" \
  --dtype "${DTYPE:-bfloat16}" \
  --attn-implementation "${ATTN_IMPLEMENTATION:-sdpa}" \
  --max-new-tokens "${MAX_NEW_TOKENS:-2048}" \
  --seed "${SEED:-20260716}" \
  "$@"
