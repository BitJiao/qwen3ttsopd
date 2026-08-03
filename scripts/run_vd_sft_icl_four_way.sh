#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
export PYTHONPATH="${ROOT}:${QWEN3_TTS_REPO:-${ROOT}/../Qwen3-TTS-main}:${PYTHONPATH:-}"

"${PYTHON:-${ROOT}/../Qwen3-TTS-main/.venv/bin/python}" -m qwen3opsd.four_way_eval \
  --comparison-manifest "${COMPARISON_MANIFEST:-outputs/emotiontalk_vd_sft_icl_three_way/manifest.jsonl}" \
  --teacher-model-path "${TEACHER_MODEL_PATH:?set TEACHER_MODEL_PATH to the Base checkpoint}" \
  --output-dir "${OUTPUT_DIR:-outputs/emotiontalk_vd_sft_icl_four_way}" \
  --device "${DEVICE:-cuda:2}" \
  "$@"
