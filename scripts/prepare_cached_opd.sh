#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

: "${INPUT_JSONL:?set INPUT_JSONL to your cached codes JSONL}"
: "${OUTPUT_JSONL:?set OUTPUT_JSONL to the paired OPD JSONL}"

COMMAND=("${PYTHON:-${ROOT}/.venv/bin/python}" -m qwen3opsd.prepare_cached_opd
  --input-jsonl "${INPUT_JSONL}"
  --output-jsonl "${OUTPUT_JSONL}")

if [[ -n "${SPEAKER_FIELD:-}" ]]; then
  COMMAND+=(--speaker-field "${SPEAKER_FIELD}")
fi

"${COMMAND[@]}" "$@"
