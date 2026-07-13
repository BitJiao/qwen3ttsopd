#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
UV_BIN="${UV_BIN:-$(command -v uv || true)}"

cd "${ROOT}"
if [[ ! -x .venv/bin/python ]] || ! .venv/bin/python -m pip --version >/dev/null 2>&1; then
  if [[ -n "${UV_BIN}" ]]; then
    "${UV_BIN}" venv --clear --seed --python "${PYTHON_BIN}" .venv
  else
    "${PYTHON_BIN}" -m venv --clear .venv
  fi
fi

.venv/bin/python -m pip install --upgrade pip setuptools wheel
.venv/bin/python -m pip install --index-url "${TORCH_INDEX_URL}" torch==2.9.0 torchaudio==2.9.0
.venv/bin/python -m pip install -r requirements.txt

QWEN3_TTS_REPO="${QWEN3_TTS_REPO:-${ROOT}/../Qwen3-TTS-main}"
if [[ ! -f "${QWEN3_TTS_REPO}/pyproject.toml" ]]; then
  echo "Qwen3-TTS source checkout not found: ${QWEN3_TTS_REPO}" >&2
  exit 2
fi
.venv/bin/python -m pip install -e "${QWEN3_TTS_REPO}"
.venv/bin/python -m pip install -e "${ROOT}"

.venv/bin/python - <<'PY'
import torch
import qwen_tts
import qwen3opsd

print(f"torch={torch.__version__} cuda={torch.version.cuda} available={torch.cuda.is_available()}")
print(f"qwen_tts={qwen_tts.__file__}")
print(f"qwen3opsd={qwen3opsd.__version__}")
PY
