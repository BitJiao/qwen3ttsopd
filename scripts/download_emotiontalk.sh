#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-${ROOT}/data/raw/emotiontalk}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
HF_BIN="${HF_BIN:-${ROOT}/.venv/bin/hf}"
METADATA_REPO="${DATA_ROOT}/metadata"
ARCHIVE_DIR="${DATA_ROOT}/archives"
EXTRACT_DIR="${DATA_ROOT}/extracted"

mkdir -p "${DATA_ROOT}" "${ARCHIVE_DIR}" "${EXTRACT_DIR}"
if [[ ! -d "${METADATA_REPO}/.git" ]]; then
  git clone --depth 1 https://github.com/NKU-HLT/EmotionTalk.git "${METADATA_REPO}"
else
  git -C "${METADATA_REPO}" pull --ff-only
fi

if [[ "${METADATA_ONLY:-0}" == "1" ]]; then
  echo "Metadata ready: ${METADATA_REPO}/EmotionTalk/dataset/mm-process"
  exit 0
fi

if [[ ! -x "${HF_BIN}" ]]; then
  echo "Hugging Face CLI not found: ${HF_BIN}; run scripts/setup_env.sh first" >&2
  exit 2
fi

export HF_ENDPOINT
"${HF_BIN}" download BAAI/Emotiontalk Audio.tar Text.tar \
  --repo-type dataset \
  --local-dir "${ARCHIVE_DIR}"

for archive in Audio.tar Text.tar; do
  marker="${EXTRACT_DIR}/.${archive}.done"
  if [[ ! -f "${marker}" ]]; then
    tar -xf "${ARCHIVE_DIR}/${archive}" -C "${EXTRACT_DIR}"
    touch "${marker}"
  fi
done

echo "EmotionTalk audio: ${EXTRACT_DIR}"
echo "EmotionTalk metadata: ${METADATA_REPO}/EmotionTalk/dataset/mm-process"
