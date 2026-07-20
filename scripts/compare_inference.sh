#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
export PYTHONPATH="${ROOT}:${QWEN3_TTS_REPO:-${ROOT}/../Qwen3-TTS-main}:${PYTHONPATH:-}"

: "${STUDENT_MODEL_PATH:?set STUDENT_MODEL_PATH to the trained VoiceDesign student}"
: "${BASE_TEACHER_MODEL_PATH:?set BASE_TEACHER_MODEL_PATH to the Base ICL candidate}"
: "${VD_TEACHER_MODEL_PATH:?set VD_TEACHER_MODEL_PATH to the frozen VoiceDesign candidate}"
: "${INPUT_JSONL:?set INPUT_JSONL to OPD validation JSONL (audio_codes are not required)}"

"${PYTHON:-${ROOT}/.venv/bin/python}" -m qwen3opsd.compare_inference \
  --student_model_path "${STUDENT_MODEL_PATH}" \
  --base_teacher_model_path "${BASE_TEACHER_MODEL_PATH}" \
  --vd_teacher_model_path "${VD_TEACHER_MODEL_PATH}" \
  --input_jsonl "${INPUT_JSONL}" \
  --output_dir "${OUTPUT_DIR:-results/inference_comparison}" \
  --device "${DEVICE:-cuda:0}" \
  --dtype "${DTYPE:-bf16}" \
  --attn_implementation "${ATTN_IMPLEMENTATION:-sdpa}" \
  --max_samples "${MAX_SAMPLES:--1}" \
  --seed "${SEED:-1234}" \
  --max_new_tokens "${MAX_NEW_TOKENS:-2048}" \
  "$@"
