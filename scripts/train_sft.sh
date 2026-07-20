#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
export PYTHONPATH="${ROOT}:${QWEN3_TTS_REPO:-${ROOT}/../Qwen3-TTS-main}:${PYTHONPATH:-}"

VOICE_DESIGN_MODEL="${VOICE_DESIGN_MODEL_PATH:-${MODEL_PATH:-}}"
: "${VOICE_DESIGN_MODEL:?set VOICE_DESIGN_MODEL_PATH (or MODEL_PATH) to a Qwen3-TTS VoiceDesign checkpoint}"

"${PYTHON:-${ROOT}/.venv/bin/python}" -m qwen3opsd.train_sft \
  --init-model-path "${VOICE_DESIGN_MODEL}" \
  --train-jsonl "${TRAIN_JSONL:-data/processed/emotiontalk/sft_train_with_codes.jsonl}" \
  --output-dir "${OUTPUT_DIR:-checkpoints/emotiontalk_sft}" \
  --batch-size "${BATCH_SIZE:-1}" \
  --gradient-accumulation-steps "${GRAD_ACCUM_STEPS:-4}" \
  --num-epochs "${NUM_EPOCHS:-1}" \
  --max-steps "${MAX_STEPS:--1}" \
  --lr "${LR:-2e-6}" \
  --save-freq "${SAVE_FREQ:-500}" \
  "$@"
