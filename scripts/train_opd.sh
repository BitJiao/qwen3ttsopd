#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

export PYTHONPATH="${ROOT}:${QWEN3_TTS_REPO:-}:${PYTHONPATH:-}"

STUDENT_MODEL="${STUDENT_MODEL_PATH:-${VOICE_DESIGN_MODEL_PATH:-${MODEL_PATH:-}}}"
TEACHER_MODE="${TEACHER_MODE:-base_icl}"
case "${TEACHER_MODE}" in
  base_icl)
    TEACHER_MODEL="${TEACHER_MODEL_PATH:-${BASE_TEACHER_MODEL_PATH:-}}"
    ;;
  voice_design)
    TEACHER_MODEL="${TEACHER_MODEL_PATH:-${VD_TEACHER_MODEL_PATH:-}}"
    ;;
  *)
    echo "TEACHER_MODE must be base_icl or voice_design, got: ${TEACHER_MODE}" >&2
    exit 1
    ;;
esac
: "${STUDENT_MODEL:?set STUDENT_MODEL_PATH to a Base or VoiceDesign checkpoint}"
: "${TEACHER_MODEL:?set TEACHER_MODEL_PATH for the selected TEACHER_MODE}"

"${PYTHON:-${ROOT}/.venv/bin/python}" -m qwen3opsd.train_opd \
  --student_model_path "${STUDENT_MODEL}" \
  --teacher_model_path "${TEACHER_MODEL}" \
  --teacher_mode "${TEACHER_MODE}" \
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
