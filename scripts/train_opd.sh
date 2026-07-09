#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

export PYTHONPATH="${ROOT}:${QWEN3_TTS_REPO:-}:${PYTHONPATH:-}"

python -m qwen3tts_opd.train_opd \
  --model_path "${MODEL_PATH:?set MODEL_PATH}" \
  --ref_model_path "${REF_MODEL_PATH:-${MODEL_PATH}}" \
  --pair_jsonl "${PAIR_JSONL:-data/opd/pairs.jsonl}" \
  --output_dir "${OUTPUT_DIR:-checkpoints/qwen3_tts_instruction_opd}" \
  --device "${DEVICE:-cuda:0}" \
  --dtype "${DTYPE:-bf16}" \
  --attn_implementation "${ATTN_IMPLEMENTATION:-sdpa}" \
  --num_epochs "${NUM_EPOCHS:-1}" \
  --max_steps "${MAX_STEPS:--1}" \
  --lr "${LR:-1e-6}" \
  --beta "${DPO_BETA:-0.1}" \
  --sft_weight "${SFT_WEIGHT:-0.2}" \
  --save_freq "${SAVE_FREQ:-100}" \
  "$@"

