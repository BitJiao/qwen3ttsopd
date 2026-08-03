#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
export PYTHONPATH="${ROOT}:${QWEN3_TTS_REPO:-${ROOT}/../Qwen3-TTS-main}:${PYTHONPATH:-}"

"${PYTHON:-${ROOT}/../Qwen3-TTS-main/.venv/bin/python}" -m qwen3opsd.gap_eval \
  --input-jsonl "${INPUT_JSONL:-data/processed/emotiontalk/sft_test.jsonl}" \
  --student-model-path "${STUDENT_MODEL_PATH:-${ROOT}/../Qwen3-TTS-12Hz-1.7B-VoiceDesign}" \
  --teacher-model-path "${TEACHER_MODEL_PATH:-${ROOT}/../Qwen3-TTS-12Hz-1.7B-Base}" \
  --output-dir "${OUTPUT_DIR:-outputs/emotiontalk_vd_icl_hard_gap}" \
  --student-device "${STUDENT_DEVICE:-cuda:1}" \
  --teacher-device "${TEACHER_DEVICE:-cuda:2}" \
  --max-new-tokens "${MAX_NEW_TOKENS:-1024}" \
  --seed "${SEED:-20260723}" \
  "$@"
