# Qwen3-TTS VoiceDesign SFT + Base-ICL Teacher OPD

本仓库实现以 **Qwen3-TTS VoiceDesign 为 student、Qwen3-TTS Base 为 ICL teacher** 的两阶段训练：

1. **Instruction SFT**：VoiceDesign student 通过官方独立 `instruct_ids` 输入自然语言音色/风格描述和目标文本。
2. **On-Policy Distillation (OPD)**：student 用自己的条件采样 codec trajectory；冻结 teacher 在相同 trajectory 上额外读取同场景参考音频和转写（ICL），逐 token 蒸馏给 student。

```text
student: VoiceDesign(instruction + target text)
teacher A: Base(target text + teacher_ref_audio/text ICL)
teacher B: frozen VoiceDesign(instruction + target text, no ICL)
action:  student 自己采样的 16-codebook codec trajectory
loss:    first-codebook/EOS KL + sub-codebook KL + small student CE
```

这里没有 ASR reward、chosen/rejected 数据、DPO loss 或基于 caption 相似度的配对筛选。

## 当前验证状态

截至 2026-07-13：

| 项目 | 结果 |
|---|---|
| EmotionTalk 官方 metadata | 已下载并解析 19,250 条 |
| 转换逻辑 | VoiceDesign student 不预留 enrollment；需重新运行全量转换审计 |
| 单元测试 | 覆盖数据转换、conditioning、JSONL 契约和 codec 时间对齐 |
| 模型级验证 | Base-ICL / VD-no-ICL 两种 teacher 组合待在 GPU 环境重新 smoke |
| 旧 Base-student smoke | 不再代表当前实现，旧 loss 记录已作废 |
| 真实 EmotionTalk 音频训练 | **未完成：Hugging Face gated access 尚未批准** |

`BAAI/Emotiontalk` 的 `Audio.tar` 为 gated 文件；必须先由数据使用者在网页接受 CC BY-NC-SA 4.0 条款。切换到 VoiceDesign student 后必须重新执行模型级 SFT/OPD smoke，旧 Base-student 结果不能作为验证依据。

## 目录结构

```text
qwen3opsd/
  emotiontalk.py       EmotionTalk 转换、循环配对和审计
  prepare_codes.py     提取 Qwen3-TTS 12 Hz target codec
  data_contract.py     VoiceDesign SFT JSONL 结构校验
  sft_dataset.py       VoiceDesign instruction SFT dataset
  train_sft.py         VoiceDesign SFT trainer
  qualify_teachers.py  student/Base-ICL/VD 三方 target-NLL 比较
  compare_inference.py student/Base-ICL/VD 三方生成推理对比
  train_opd.py         OPD 入口
  infer.py             SFT/OPD checkpoint 推理
qwen3tts_opd/
  conditioning.py      VoiceDesign student / Base teacher 条件输入契约
  teacher_modes.py     Base-ICL / VoiceDesign teacher 模式定义
  core.py              token replay、teacher/student logits、KL、保存
  train_opd.py         OPD 训练实现（保留旧包兼容）
scripts/
  setup_env.sh
  download_emotiontalk.sh
  convert_emotiontalk.sh
  prepare_sft.sh
  train_sft.sh
  qualify_teachers.sh
  compare_inference.sh
  train_opd.sh
tests/
```

数据、checkpoint、WAV 和 Hugging Face token 均由 `.gitignore` 排除，不会推送到 GitHub。

## 1. 环境

要求：

- Linux、Python 3.10+
- NVIDIA GPU；完整训练建议 40 GB 以上显存
- Qwen3-TTS 官方源码 checkout
- Qwen3-TTS-12Hz-1.7B-VoiceDesign student 权重
- Qwen3-TTS-12Hz-1.7B-Base teacher 权重（提供 ICL 参考音频条件）
- 独立的冻结 VoiceDesign teacher checkpoint（无 ICL 候选；不能与 student 完全相同）
- `ffmpeg`、SoX

```bash
git clone git@github.com:BitJiao/qwen3ttsopd.git
cd qwen3ttsopd

export QWEN3_TTS_REPO=/absolute/path/Qwen3-TTS
export TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128
bash scripts/setup_env.sh
```

`setup_env.sh` 安装本仓库实测版本：PyTorch/Torchaudio 2.9.0、Transformers 4.57.3、Accelerate 1.12.0，并 editable-install 官方 Qwen3-TTS 与本项目。CUDA 版本不同的机器需要修改 `TORCH_INDEX_URL`；PyTorch 与 Torchaudio 的版本必须一致。

每次运行前：

```bash
source .venv/bin/activate
export QWEN3_TTS_REPO=/absolute/path/Qwen3-TTS
export PYTHONPATH="$(pwd):${QWEN3_TTS_REPO}:${PYTHONPATH:-}"
```

## 2. 下载 EmotionTalk

数据许可证是 **CC BY-NC-SA 4.0，仅限非商业用途**。

1. 登录 Hugging Face。
2. 打开 <https://huggingface.co/datasets/BAAI/Emotiontalk>。
3. 点击申请/同意访问条款，等待账号进入 authorized list。
4. 在服务器登录同一个账号。

```bash
hf auth login
bash scripts/download_emotiontalk.sh
```

国内网络可指定镜像：

```bash
HF_ENDPOINT=https://hf-mirror.com bash scripts/download_emotiontalk.sh
```

脚本会下载：

- `Audio.tar`：约 14.8 GB，训练所需 WAV。
- `Text.tar`：原始文本标注。
- `NKU-HLT/EmotionTalk` metadata 仓库：`transcription.csv` 和 `audio.csv`。

默认位置：

```text
data/raw/emotiontalk/
  archives/{Audio.tar,Text.tar}
  extracted/
  metadata/EmotionTalk/dataset/mm-process/
```

只下载公开 metadata、暂不下载 gated 音频：

```bash
METADATA_ONLY=1 bash scripts/download_emotiontalk.sh
```

遇到 `403 GatedRepo` 说明当前 token 对应账号尚未获授权，换镜像不能绕过授权。

## 3. EmotionTalk 分组规则

真实 key 形如：

```text
G00009/G00009_42/G00009_42_14/G00009_42_14_024
```

解析为：

- split group：`G00009`
- scene/dialogue：`G00009/G00009_42`
- speaker：`14`
- utterance sequence：`024`

官方 split 按第一层 group：

- validation：`G00001`、`G00012`
- test：`G00003`、`G00015`
- 其余为 train

VoiceDesign student 不读取 enrollment，因此转换器不会再为每个 speaker 预留并丢弃一条音频。所有合法音频都可用于 SFT；OPD teacher reference 仍严格限制在同 speaker、同 scene 内。

OPD 严格按 `(speaker_id, scene_id)` 分组：

```text
自然顺序: 001, 002, 010
循环关系: 001 -> 002, 002 -> 010, 010 -> 001
```

不使用风格词权重、caption 相似度或阈值。以下组会报错或跳过：

- 单条组
- 重复 sequence
- 重复音频路径或内容
- 相邻循环 pair 的文本相同
- target/reference 自引用

`group_audit.jsonl` 记录每个组的完整循环或跳过原因。转换结束还会验证：每条有效 target 音频恰好作为 teacher reference 一次。

## 4. 转换数据

```bash
bash scripts/convert_emotiontalk.sh
```

等价的完整命令：

```bash
python -m qwen3opsd.emotiontalk \
  --transcription-csv data/raw/emotiontalk/metadata/EmotionTalk/dataset/mm-process/transcription.csv \
  --caption-csv data/raw/emotiontalk/metadata/EmotionTalk/dataset/mm-process/audio.csv \
  --audio-root data/raw/emotiontalk/extracted \
  --output-dir data/processed/emotiontalk \
  --caption-key caption_1 \
  --on-invalid-group skip \
  --check-audio \
  --check-audio-hash
```

`--check-audio-hash` 最严格但会读取全部音频；大规模转换可先只用 `--check-audio`，正式训练前再跑一次 hash 审计。

仓库早期记录的计数来自 Base student enrollment 方案，切换到 VoiceDesign 后已经失效。重新运行转换后，以新生成的 `summary.json` 和 `group_audit.jsonl` 为准；不要沿用旧的 `enrollment_rows` 或 split 数量。

输出包括：

```text
sft_{train,val,test}.jsonl
opd_{train,val,test}.jsonl
group_audit.jsonl
summary.json
```

核心字段：

| 字段 | 消费方 | 含义 |
|---|---|---|
| `text` | student + teacher | 当前 target 转写 |
| `instruction` | student + teacher | 当前 target 的综合语音 caption |
| `target_audio` / `audio` | SFT | 当前要学习的音频 |
| `teacher_ref_audio` | OPD teacher | 同 speaker、同 scene 的下一条音频 |
| `teacher_ref_text` | OPD teacher | teacher reference 的准确转写 |

`ref_audio/ref_text` 仅为兼容旧入口；新代码优先读取语义明确的字段。

### 4.1 训练 JSONL 的准确结构

训练文件是 **JSONL**，不是一个 JSON 数组：每一行必须是一个完整 JSON object，行尾不加逗号，文件外层也不加 `[` / `]`。仓库脚本先 `cd` 到仓库根目录，因此相对音频路径默认相对于仓库根目录；在其他目录直接运行 Python 时建议使用绝对路径。

SFT 提取 codec 前的输入，例如 `sft_train.jsonl`：

```jsonl
{"text":"今天很高兴见到你。","instruction":"女性声音清亮自然，语速稍快。","target_audio":"data/audio/target_001.wav","language":"Chinese"}
{"text":"我们明天再讨论这个问题。","instruction":"语气平静、自然。","target_audio":"data/audio/target_002.wav","language":"Chinese"}
```

`target_audio` 是推荐字段；旧数据可以用 `audio` 代替。VoiceDesign SFT 不需要 `student_spk_audio/ref_audio`。运行 `scripts/prepare_sft.sh` 后，每行会保留原字段并增加 `audio_codes`：

```jsonl
{"text":"今天很高兴见到你。","instruction":"女性声音清亮自然，语速稍快。","target_audio":"data/audio/target_001.wav","language":"Chinese","audio_codes":[[101,202,303,404,505,606,707,808,909,1001,1102,1203,1304,1405,1506,1607],[102,203,304,405,506,607,708,809,910,1002,1103,1204,1305,1406,1507,1608]]}
```

这里 `audio_codes` 的形状必须是 `[T, 16]`：外层长度 `T` 是 12 Hz codec 帧数，每一帧恰好包含 16 个 codebook token。示例只写了两帧用于展示结构，真实音频会有更多帧。`TRAIN_JSONL` 必须指向这个带 `audio_codes` 的文件。

OPD 的 `INPUT_JSONL` 不需要预先生成 `audio_codes`，因为 trajectory 由 student 在线采样。每行推荐结构如下：

```jsonl
{"text":"今天很高兴见到你。","instruction":"女性声音清亮自然，语速稍快。","target_audio":"data/audio/target_001.wav","teacher_ref_audio":"data/audio/same_scene_002.wav","teacher_ref_text":"这是一条同场景参考语音的准确转写。","language":"Chinese"}
```

OPD 始终需要 `text`（或 `target_text`）；`base_icl` 模式和三方 qualification 还需要 `teacher_ref_audio/teacher_ref_text`，单独运行 `voice_design` teacher 模式则不要求这两个 ICL 字段。`instruction` 允许为空但 VoiceDesign 训练通常应提供，`language` 和仅用于泄漏检查/审计的 `target_audio` 可以省略。

## 5. Instruction SFT

先提取 target audio codes：

```bash
VOICE_DESIGN_MODEL_PATH=/absolute/path/Qwen3-TTS-12Hz-1.7B-VoiceDesign \
DEVICE=cuda:0 \
BATCH_SIZE=16 \
bash scripts/prepare_sft.sh
```

然后训练：

```bash
VOICE_DESIGN_MODEL_PATH=/absolute/path/Qwen3-TTS-12Hz-1.7B-VoiceDesign \
TRAIN_JSONL=data/processed/emotiontalk/sft_train_with_codes.jsonl \
OUTPUT_DIR=checkpoints/emotiontalk_sft \
BATCH_SIZE=1 \
GRAD_ACCUM_STEPS=8 \
NUM_EPOCHS=3 \
LR=2e-6 \
bash scripts/train_sft.sh --overwrite
```

训练约束：

- checkpoint 必须是 `tts_model_type=voice_design`，不接受 Base/CustomVoice。
- instruction 使用官方独立 user prompt，不会拼进 assistant target text。
- speech tokenizer 冻结，只训练 VoiceDesign talker。
- checkpoint 保持 `tts_model_type=voice_design`，推理继续调用 `generate_voice_design`。

## 6. Teacher Qualification 与 OPD

OPD 支持两个冻结 teacher 候选：

- `base_icl`：Base teacher 读取 target text 和 `teacher_ref_audio/text`。
- `voice_design`：VD teacher 只读取与 student 相同的 instruction/target text，不使用 ICL。

先给 OPD validation 子集补 target `audio_codes`：

```bash
VOICE_DESIGN_MODEL_PATH=/absolute/path/Qwen3-TTS-12Hz-1.7B-VoiceDesign \
INPUT_JSONL=data/processed/emotiontalk/opd_val.jsonl \
OUTPUT_JSONL=data/processed/emotiontalk/opd_val_with_codes.jsonl \
bash scripts/prepare_sft.sh
```

然后顺序加载 student、Base-ICL teacher 和 VD teacher，在相同真实 target codes 上比较 codec-0、EOS、sub-codebook 和总 NLL：

```bash
STUDENT_MODEL_PATH=checkpoints/emotiontalk_sft/final \
BASE_TEACHER_MODEL_PATH=/absolute/path/Qwen3-TTS-12Hz-1.7B-Base \
VD_TEACHER_MODEL_PATH=/absolute/path/stronger-VoiceDesign-checkpoint \
INPUT_JSONL=data/processed/emotiontalk/opd_val_with_codes.jsonl \
DEVICE=cuda:0 \
MAX_SAMPLES=500 \
bash scripts/qualify_teachers.sh
```

该工具一次只在设备上保留一个模型，输出：

```text
results/teacher_qualification/scores.jsonl
results/teacher_qualification/summary.json
```

定义 `margin = student_total_nll - teacher_total_nll`。`positive_rate` 越高，teacher 在真实 target 上胜过 student 的样本比例越高。若 `same_student_and_vd_teacher_path=true`，VD teacher 与 student 是同一路径，这个比较通常没有蒸馏价值。NLL qualification 只验证 token 建模能力，正式长跑前仍应补充 WER、风格/情绪匹配和音质生成评测。

再运行三方生成推理，实际比较可懂度、音质和 instruction/情绪匹配。输入使用原始 `opd_val.jsonl` 即可，不要求 `audio_codes`：

```bash
STUDENT_MODEL_PATH=checkpoints/emotiontalk_sft/final \
BASE_TEACHER_MODEL_PATH=/absolute/path/Qwen3-TTS-12Hz-1.7B-Base \
INPUT_JSONL=data/processed/emotiontalk/opd_val.jsonl \
OUTPUT_DIR=results/inference_comparison \
DEVICE=cuda:0 \
MAX_SAMPLES=100 \
bash scripts/compare_inference.sh
```

默认只对比 VoiceDesign student 和 Base-ICL teacher。若还要加入第三个无 ICL 的 VD teacher，再设置 `VD_TEACHER_MODEL_PATH=/absolute/path/stronger-VoiceDesign-checkpoint`。脚本按顺序加载模型，单卡同一时刻只保留一个模型；每条样本的各方推理使用相同 seed 和相同采样参数：

```text
student:    VoiceDesign student + instruction/text
base_icl:   Base teacher + teacher_ref_audio/text ICL + target text
vd_teacher: frozen VoiceDesign teacher + instruction/text（无 ICL）
```

输出结构：

```text
results/inference_comparison/
  audio/student/*.wav
  audio/base_icl/*.wav
  audio/vd_teacher/*.wav  # 仅设置 VD_TEACHER_MODEL_PATH 时生成
  manifest.jsonl
  run_config.json
  summary.json
```

`manifest.jsonl` 将同一条样本的 target、instruction、ICL reference 和三份生成音频对应起来，可直接交给 ASR/WER、情绪分类器或人工盲听。中断后用完全相同的参数重跑会跳过已有 wav；参数发生变化时必须改 `OUTPUT_DIR` 或显式传 `--overwrite`，避免混合两次实验。

使用 Base-ICL teacher 训练：

```bash
STUDENT_MODEL_PATH=checkpoints/emotiontalk_sft/final \
TEACHER_MODEL_PATH=/absolute/path/Qwen3-TTS-12Hz-1.7B-Base \
TEACHER_MODE=base_icl \
INPUT_JSONL=data/processed/emotiontalk/opd_train.jsonl \
OUTPUT_DIR=checkpoints/emotiontalk_opd \
DEVICE=cuda:0 \
TEACHER_DEVICE=cuda:1 \
NUM_EPOCHS=1 \
LR=1e-6 \
MAX_NEW_TOKENS=2048 \
KL_TEMPERATURE=1.0 \
SUB_KL_WEIGHT=0.3 \
STUDENT_CE_WEIGHT=0.05 \
bash scripts/train_opd.sh --shuffle --overwrite
```

使用无 ICL 的 VD teacher 时，只替换 teacher checkpoint 和模式：

```bash
STUDENT_MODEL_PATH=checkpoints/emotiontalk_sft/final \
TEACHER_MODEL_PATH=/absolute/path/stronger-VoiceDesign-checkpoint \
TEACHER_MODE=voice_design \
INPUT_JSONL=data/processed/emotiontalk/opd_train.jsonl \
OUTPUT_DIR=checkpoints/emotiontalk_opd_vd_teacher \
DEVICE=cuda:0 \
TEACHER_DEVICE=cuda:1 \
bash scripts/train_opd.sh --shuffle --overwrite
```

单张 80 GB GPU 也可以把 `DEVICE` 和 `TEACHER_DEVICE` 都设为 `cuda:0`；两张 GPU 会降低训练时的单卡显存压力。

每一步严格执行：

1. VoiceDesign student 读取独立 instruction 和 target text，采样 codec codes。
2. 所选 teacher 按模式读取 Base ICL 条件或 VD instruction/text 条件。
3. teacher 和 student 都 replay **同一条 student codes**。
4. 优化首 codebook KL、子 codebook KL 和小权重 student token CE。

## 7. 推理

SFT 和 OPD 输出仍是 VoiceDesign checkpoint，推理时直接提供文字 instruction：

```bash
python -m qwen3opsd.infer \
  --model-path checkpoints/emotiontalk_opd/final \
  --instruction "女性声音清亮自然，语速稍快，语气轻松。" \
  --text "今天很高兴见到你。" \
  --output-wav outputs/demo.wav \
  --device cuda:0
```

训练和推理统一使用官方 VoiceDesign 消息结构：

```text
user: <instruction>
assistant: <target transcript>
```

不要把 instruction 手工拼进 `text`；训练和推理都会调用官方 `_build_instruct_text`，将它作为独立 user message。

## 8. 测试与短程检查

```bash
python -m unittest discover -s tests -v
python -m compileall -q qwen3opsd qwen3tts_opd tests
```

正式长跑前建议：

```bash
# SFT one step
MAX_STEPS=1 SAVE_FREQ=0 bash scripts/train_sft.sh --overwrite

# OPD one step，缩短 rollout
MAX_STEPS=1 MAX_NEW_TOKENS=32 SAVE_FREQ=0 bash scripts/train_opd.sh --overwrite
```

确认 loss、grad norm、checkpoint reload 均正常，再移除 `MAX_STEPS`。

## 9. 常见问题

**`403 GatedRepo`**

当前 Hugging Face 账号尚未在 EmotionTalk authorized list。必须在数据集网页接受条款，token 或镜像本身不能绕过。

**`check_model_inputs() missing ... func`**

Transformers 版本不匹配。使用 `transformers==4.57.3`。

**模型类型报错**

SFT/student 必须使用 `Qwen3-TTS-12Hz-1.7B-VoiceDesign`。`TEACHER_MODE=base_icl` 要求 Base checkpoint；`TEACHER_MODE=voice_design` 要求 VoiceDesign checkpoint。VoiceDesign 没有 speaker encoder，因此只能作为无 ICL teacher。

**OPD 显存不足**

把所选 teacher 放到另一张 GPU；减小 `MAX_NEW_TOKENS`。当前实现是一条样本一次 on-policy rollout，不使用数据 batch。

**某些组被跳过**

查看 `group_audit.jsonl` 的 `reason`。不要为了增加数量关闭 self-reference、同音频或同文本检查，否则 teacher 条件可能直接泄漏 target。

## 许可证

本仓库代码遵循仓库自身许可证。Qwen3-TTS 权重与源码遵循其官方许可证。EmotionTalk 数据遵循 **CC BY-NC-SA 4.0**；使用者负责完成授权、署名、非商业限制和 ShareAlike 要求。
