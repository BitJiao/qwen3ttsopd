#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
export PYTHONPATH="${ROOT}:${QWEN3_TTS_REPO:-${ROOT}/../Qwen3-TTS-main}:${PYTHONPATH:-}"

: "${STUDENT_MODEL_PATH:?set STUDENT_MODEL_PATH to the trainable VoiceDesign checkpoint}"
: "${BASE_TEACHER_MODEL_PATH:?set BASE_TEACHER_MODEL_PATH to the Base ICL candidate}"
: "${VD_TEACHER_MODEL_PATH:?set VD_TEACHER_MODEL_PATH to the frozen VoiceDesign candidate}"
: "${INPUT_JSONL:?set INPUT_JSONL to OPD JSONL with target audio_codes}"

"${PYTHON:-${ROOT}/.venv/bin/python}" -m qwen3opsd.qualify_teachers \
  --student_model_path "${STUDENT_MODEL_PATH}" \
  --base_teacher_model_path "${BASE_TEACHER_MODEL_PATH}" \
  --vd_teacher_model_path "${VD_TEACHER_MODEL_PATH}" \
  --input_jsonl "${INPUT_JSONL}" \
  --output_jsonl "${OUTPUT_JSONL:-results/teacher_qualification/scores.jsonl}" \
  --summary_json "${SUMMARY_JSON:-results/teacher_qualification/summary.json}" \
  --device "${DEVICE:-cuda:0}" \
  --dtype "${DTYPE:-bf16}" \
  --attn_implementation "${ATTN_IMPLEMENTATION:-sdpa}" \
  --sub_loss_weight "${SUB_LOSS_WEIGHT:-0.3}" \
  --max_samples "${MAX_SAMPLES:--1}" \
  "$@"
