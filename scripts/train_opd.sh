#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

export PYTHONPATH="${ROOT}:${QWEN3_TTS_REPO:-}:${PYTHONPATH:-}"

"${PYTHON:-${ROOT}/.venv/bin/python}" -m qwen3opsd.train_opd \
  --student_model_path "${STUDENT_MODEL_PATH:-${MODEL_PATH:?set MODEL_PATH or STUDENT_MODEL_PATH}}" \
  --teacher_model_path "${TEACHER_MODEL_PATH:-${STUDENT_MODEL_PATH:-${MODEL_PATH}}}" \
  --input_jsonl "${INPUT_JSONL:?set INPUT_JSONL}" \
  --output_dir "${OUTPUT_DIR:-checkpoints/qwen3_tts_opd}" \
  --device "${DEVICE:-cuda:0}" \
  --teacher_device "${TEACHER_DEVICE:-${DEVICE:-cuda:0}}" \
  --dtype "${DTYPE:-bf16}" \
  --teacher_dtype "${TEACHER_DTYPE:-${DTYPE:-bf16}}" \
  --attn_implementation "${ATTN_IMPLEMENTATION:-sdpa}" \
  --num_epochs "${NUM_EPOCHS:-1}" \
  --max_steps "${MAX_STEPS:--1}" \
  --lr "${LR:-1e-6}" \
  --kl_temperature "${KL_TEMPERATURE:-1.0}" \
  --sub_kl_weight "${SUB_KL_WEIGHT:-0.3}" \
  --student_ce_weight "${STUDENT_CE_WEIGHT:-0.05}" \
  --save_freq "${SAVE_FREQ:-100}" \
  "$@"
