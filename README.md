# Qwen3-TTS On-Policy Distillation

On-policy distillation tools for Qwen3-TTS Base.

This repo trains a student on its own sampled audio-code trajectories. A frozen teacher
scores the same student trajectory token by token while seeing privileged reference
information.

```text
Student condition: text + instruction + speaker embedding
Teacher condition: text + instruction + ref_audio/ref_text ICL
Trajectory:        sampled by the student
Loss:              KL(teacher next-code distribution || student next-code distribution)
```

No ASR reward, chosen/rejected pair building, or DPO loss is used.

## Install

```bash
git clone <this-repo>
cd qwen3tts-opd

python3.11 -m venv .venv
source .venv/bin/activate

pip install torch==2.3.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# Expose the official Qwen3-TTS source checkout. The OPD trainer uses its
# inference wrapper and low-level talker forward methods.
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

`ref_text` is required because the teacher uses Qwen3-TTS ICL conditioning.
Instruction keys accepted: `instruction`, `emotion_instruction`, `style_instruction`, `instruct`.

## Train OPD

```bash
MODEL_PATH=/path/to/qwen3-tts-student-or-sft-checkpoint \
TEACHER_MODEL_PATH=/path/to/frozen-teacher-checkpoint \
INPUT_JSONL=data/train_instruction.jsonl \
OUTPUT_DIR=checkpoints/qwen3_tts_opd \
NUM_EPOCHS=1 \
LR=1e-6 \
bash scripts/train_opd.sh --overwrite
```

If `TEACHER_MODEL_PATH` is omitted, the initial student checkpoint is used as the
frozen teacher. This is useful for privileged-information OPD where the teacher
is not a larger model, but sees `ref_audio/ref_text` ICL while the student sees
only speaker embedding.

Useful knobs:

```bash
KL_TEMPERATURE=1.0
SUB_KL_WEIGHT=0.3
STUDENT_CE_WEIGHT=0.05
MAX_NEW_TOKENS=2048
SAVE_FREQ=100
```

Per step, the trainer:

1. Samples audio codes from the student with x-vector-only voice-clone conditioning.
2. Replays that exact student code trajectory under the frozen teacher with ICL conditioning.
3. Replays the same trajectory under the trainable student with x-vector-only conditioning.
4. Optimizes first-codebook KL plus sub-codebook KL and a small CE term on the student-sampled tokens.

## Notes

- The teacher and student must share the same Qwen3-TTS audio-code action space.
- Reference audio must be compatible with the Qwen3-TTS speaker encoder; 24 kHz wav is the safest format.
- The text seen by both policies is formatted as:

```text
Instruction: ...
Text: ...
```

Change this in `qwen3tts_opd/instruction_utils.py` if your checkpoint expects a different control template.
