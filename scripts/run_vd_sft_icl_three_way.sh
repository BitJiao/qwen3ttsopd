#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
export PYTHONPATH="${ROOT}:${QWEN3_TTS_REPO:-${ROOT}/../Qwen3-TTS-main}:${PYTHONPATH:-}"

"${PYTHON:-${ROOT}/../Qwen3-TTS-main/.venv/bin/python}" -m qwen3opsd.three_way_eval \
  --baseline-manifest "${BASELINE_MANIFEST:-outputs/emotiontalk_vd_icl_hard_gap/manifest.jsonl}" \
  --sft-model-path "${SFT_MODEL_PATH:?set SFT_MODEL_PATH to the trained VoiceDesign checkpoint}" \
  --output-dir "${OUTPUT_DIR:-outputs/emotiontalk_vd_sft_icl_three_way}" \
  --device "${DEVICE:-cuda:2}" \
  "$@"
