#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
export PYTHONPATH="${ROOT}:${QWEN3_TTS_REPO:-${ROOT}/../Qwen3-TTS-main}:${PYTHONPATH:-}"

"${PYTHON:-${ROOT}/../Qwen3-TTS-main/.venv/bin/python}" -m qwen3opsd.train_vd_sft \
  --init-model-path "${MODEL_PATH:?set MODEL_PATH to a VoiceDesign checkpoint}" \
  --train-jsonl "${TRAIN_JSONL:-data/processed/emotiontalk/sft_train_with_codes.jsonl}" \
  --output-dir "${OUTPUT_DIR:-checkpoints/emotiontalk_vd_sft}" \
  --batch-size "${BATCH_SIZE:-1}" \
  --gradient-accumulation-steps "${GRAD_ACCUM_STEPS:-8}" \
  --num-epochs "${NUM_EPOCHS:-3}" \
  --max-steps "${MAX_STEPS:--1}" \
  --lr "${LR:-2e-6}" \
  --save-freq "${SAVE_FREQ:-0}" \
  --log-every "${LOG_EVERY:-10}" \
  "$@"
