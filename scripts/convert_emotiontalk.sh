#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-${ROOT}/data/raw/emotiontalk}"
METADATA_DIR="${METADATA_DIR:-${DATA_ROOT}/metadata/EmotionTalk/dataset/mm-process}"
AUDIO_ROOT="${AUDIO_ROOT:-${DATA_ROOT}/extracted}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/data/processed/emotiontalk}"

cd "${ROOT}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
"${PYTHON:-${ROOT}/.venv/bin/python}" -m qwen3opsd.emotiontalk \
  --transcription-csv "${METADATA_DIR}/transcription.csv" \
  --caption-csv "${METADATA_DIR}/audio.csv" \
  --audio-root "${AUDIO_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --caption-key "${CAPTION_KEY:-caption_1}" \
  --on-invalid-group "${ON_INVALID_GROUP:-skip}" \
  --check-audio \
  "$@"
