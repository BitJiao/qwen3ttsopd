# Qwen3-TTS Instruction OPD

Standalone tools for instruction-controlled OPD/DPO on Qwen3-TTS Base.

The intended setup is asymmetric:

- Teacher: target text + emotion/style instruction + reference audio/text, using ICL continuation.
- Student: target text + emotion/style instruction + speaker embedding only.
- OPD: sample candidates, score them, write chosen/rejected pairs, then train with DPO plus a small chosen NLL term.

## Install

```bash
git clone <this-repo>
cd qwen3tts-opd

python3.11 -m venv .venv
source .venv/bin/activate

pip install torch==2.3.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# Install or expose the official Qwen3-TTS source/package.
# If using a local checkout:
export QWEN3_TTS_REPO=/path/to/Qwen3-TTS
export PYTHONPATH="$(pwd):${QWEN3_TTS_REPO}:${PYTHONPATH:-}"
```

`ffmpeg` is required by the audio stack:

```bash
apt-get update && apt-get install -y ffmpeg
```

## Input Data

JSONL rows:

```json
{
  "sample_id": "utt_0001",
  "text": "今晚的风比昨天更冷。",
  "instruction": "用压抑、悲伤、语速偏慢的语气朗读。",
  "ref_audio": "/abs/path/ref.wav",
  "ref_text": "这是参考音频的转写。",
  "language": "Chinese"
}
```

Instruction keys accepted: `instruction`, `emotion_instruction`, `style_instruction`, `instruct`.

## Build Preference Pairs

```bash
MODEL_PATH=/path/to/Qwen3-TTS-12Hz-1.7B-Base \
INPUT_JSONL=data/train_instruction.jsonl \
OUTPUT_JSONL=data/opd/pairs.jsonl \
AUDIO_DIR=data/opd/audio \
GROUP_SIZE=4 \
REWARD_ASR_BACKEND=none \
bash scripts/build_pairs.sh --overwrite
```

Default reward is `qwen3tts_opd.reward.wer_sim_reward:compute_score`.
With `REWARD_ASR_BACKEND=none`, WER is disabled and the reward is mostly MFCC speaker similarity.
For content scoring, set:

```bash
REWARD_ASR_BACKEND=transformers
ASR_MODEL_PATH=/path/to/openai-whisper-small
ASR_DEVICE_INDEX=0
```

Pair output contains `chosen_codes` and `rejected_codes`, so OPD training does not need to decode audio again.

## Train OPD

```bash
MODEL_PATH=/path/to/sft-or-base-checkpoint \
REF_MODEL_PATH=/path/to/frozen-reference-checkpoint \
PAIR_JSONL=data/opd/pairs.jsonl \
OUTPUT_DIR=checkpoints/qwen3_tts_instruction_opd \
NUM_EPOCHS=1 \
LR=1e-6 \
DPO_BETA=0.1 \
SFT_WEIGHT=0.2 \
bash scripts/train_opd.sh --overwrite
```

Loss:

```text
L = DPO(chosen, rejected; frozen_ref) + SFT_WEIGHT * NLL(chosen)
```

Best practice:

1. Generate teacher ICL audio codes and do an instruction SFT first.
2. Use the SFT checkpoint as `MODEL_PATH`.
3. Use the same SFT checkpoint frozen as `REF_MODEL_PATH`.
4. Build OPD pairs and train.

## Notes

- Reference audio must be compatible with Qwen3-TTS speaker encoder; 24 kHz wav is the safest format.
- The text seen by the model is formatted as:

```text
Instruction: ...
Text: ...
```

Change this in `qwen3tts_opd/instruction_utils.py` if your Qwen3-TTS checkpoint expects a different control template.

