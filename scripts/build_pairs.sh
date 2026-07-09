#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

export PYTHONPATH="${ROOT}:${QWEN3_TTS_REPO:-}:${PYTHONPATH:-}"
export REWARD_ASR_BACKEND=${REWARD_ASR_BACKEND:-none}
export REWARD_WER_WEIGHT=${REWARD_WER_WEIGHT:-0.6}
export REWARD_SIM_WEIGHT=${REWARD_SIM_WEIGHT:-0.4}

python -m qwen3tts_opd.build_pairs \
  --model_path "${MODEL_PATH:?set MODEL_PATH}" \
  --input_jsonl "${INPUT_JSONL:?set INPUT_JSONL}" \
  --output_jsonl "${OUTPUT_JSONL:-data/opd/pairs.jsonl}" \
  --audio_dir "${AUDIO_DIR:-data/opd/audio}" \
  --reward_fn "${REWARD_FN:-qwen3tts_opd.reward.wer_sim_reward:compute_score}" \
  --device "${DEVICE:-cuda:0}" \
  --dtype "${DTYPE:-bf16}" \
  --attn_implementation "${ATTN_IMPLEMENTATION:-sdpa}" \
  --group_size "${GROUP_SIZE:-4}" \
  --instruction_template "${INSTRUCTION_TEMPLATE:-qwen_control}" \
  --margin "${OPD_MARGIN:-0.05}" \
  "$@"

