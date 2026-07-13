#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
export PYTHONPATH="${ROOT}:${QWEN3_TTS_REPO:-${ROOT}/../Qwen3-TTS-main}:${PYTHONPATH:-}"

"${PYTHON:-${ROOT}/.venv/bin/python}" -m qwen3opsd.prepare_codes \
  --model-path "${MODEL_PATH:?set MODEL_PATH}" \
  --input-jsonl "${INPUT_JSONL:-data/processed/emotiontalk/sft_train.jsonl}" \
  --output-jsonl "${OUTPUT_JSONL:-data/processed/emotiontalk/sft_train_with_codes.jsonl}" \
  --device "${DEVICE:-cuda:0}" \
  --batch-size "${BATCH_SIZE:-16}" \
  "$@"
