#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
export PYTHONPATH="${ROOT}:${QWEN3_TTS_REPO:-${ROOT}/../Qwen3-TTS-main}:${PYTHONPATH:-}"

VOICE_DESIGN_MODEL="${VOICE_DESIGN_MODEL_PATH:-${MODEL_PATH:-}}"
: "${VOICE_DESIGN_MODEL:?set VOICE_DESIGN_MODEL_PATH (or MODEL_PATH) to a Qwen3-TTS VoiceDesign checkpoint}"

"${PYTHON:-${ROOT}/.venv/bin/python}" -m qwen3opsd.prepare_codes \
  --model-path "${VOICE_DESIGN_MODEL}" \
  --input-jsonl "${INPUT_JSONL:-data/processed/emotiontalk/sft_train.jsonl}" \
  --output-jsonl "${OUTPUT_JSONL:-data/processed/emotiontalk/sft_train_with_codes.jsonl}" \
  --device "${DEVICE:-cuda:0}" \
  --batch-size "${BATCH_SIZE:-16}" \
  "$@"
