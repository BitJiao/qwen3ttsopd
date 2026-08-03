# Qwen3-TTS Emotion SFT + Multi-Mode OPD

本仓库实现两条可对照的情感可控训练路径：严格多说话人 Base/x-vector student 主线，以及 VoiceDesign student 实验线。两者都可使用冻结 Base ICL teacher；VoiceDesign 还可使用无 ICL 的冻结 VoiceDesign teacher。

1. **Instruction SFT**：caption 通过官方独立 `instruct_ids` 输入；Base student 额外读取独立同 speaker enrollment，VoiceDesign student 不读取 enrollment。
2. **On-Policy Distillation (OPD)**：student 用自己的条件采样 codec trajectory；冻结 teacher 在相同 trajectory 上额外读取同场景参考音频和转写（ICL），逐 token 蒸馏给 student。

```text
student A: Base(instruction + target text + enrollment x-vector)
student B: VoiceDesign(instruction + target text)
teacher A: Base(target text + teacher_ref_audio/text ICL)
teacher B: frozen VoiceDesign(instruction + target text, no ICL)
action:  student 自己采样的 16-codebook codec trajectory
loss:    first-codebook/EOS KL + sub-codebook KL + small student CE
```

这里没有 ASR reward、chosen/rejected 数据、DPO loss 或基于 caption 相似度的配对筛选。

## 当前验证状态

截至 2026-08-03：

| 项目 | 结果 |
|---|---|
| EmotionTalk 数据 | gated 音频已下载，19,250 条 metadata 已全量转换 |
| 全量转换审计 | 1,452 个有效循环组 |
| 单元测试 | 43/43 通过 |
| Qwen3-TTS Base 加载 | 1.7B Base、16 codebooks，GPU 加载成功 |
| Caption SFT | 3 epochs、5,775 optimizer steps 已完成 |
| Caption OPD | 3,222 / 15,134 samples；最近 checkpoint 为 `step_3000`，需在新服务器续跑 |
| 旧版 OPD 流程 | 15,134 / 15,134 samples 已完成，仅作历史对照 |

`BAAI/Emotiontalk` 的 `Audio.tar` 是 gated 文件。即使本机已有数据，另一台服务器仍必须使用获授权的 Hugging Face 账号下载，并接受 CC BY-NC-SA 4.0 条款；仓库中的 portable manifests 不绕过该授权。

## 目录结构

```text
qwen3opsd/
  emotiontalk.py       EmotionTalk 转换、循环配对和审计
  portable_manifest.py 跨服务器导出/物化相对音频路径
  prepare_cached_opd.py 缓存 codes/spk embedding 数据的 OPD 配对
  prepare_codes.py     提取 Qwen3-TTS 12 Hz target codec
  data_contract.py     VoiceDesign SFT JSONL 结构校验
  sft_dataset.py       Base / VoiceDesign instruction SFT datasets
  train_sft.py         Base 多说话人 SFT trainer
  train_vd_sft.py      VoiceDesign SFT trainer
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
  prepare_cached_opd.sh
  train_sft.sh
  qualify_teachers.sh
  compare_inference.sh
  train_opd.sh
tests/
```

原始数据、checkpoint、WAV、预计算 codec 和 Hugging Face token 均由 `.gitignore` 排除，不会推送到 GitHub。仓库中的 `dataset_manifests/emotiontalk/` 保存可迁移的相对路径清单；完整数据盘点见 [`docs/EMOTION_DATA_INVENTORY.md`](docs/EMOTION_DATA_INVENTORY.md)。

## 1. 环境

要求：

- Linux、Python 3.10+
- NVIDIA GPU；完整训练建议 40 GB 以上显存
- Qwen3-TTS 官方源码 checkout
- Qwen3-TTS-12Hz-1.7B-Base 权重（Base student/ICL teacher）
- 可选：Qwen3-TTS-12Hz-1.7B-VoiceDesign student 权重
- 可选：独立冻结 VoiceDesign teacher checkpoint（无 ICL 候选）
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

portable manifests 按 split 和 speaker 预留独立 enrollment，供 Base student 使用；VoiceDesign student 会忽略这些 enrollment。OPD teacher reference 始终严格限制在同 speaker、同 scene 内。

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
  --student-mode base \
  --on-invalid-group skip \
  --check-audio \
  --check-audio-hash
```

`--check-audio-hash` 最严格但会读取全部音频；大规模转换可先只用 `--check-audio`，正式训练前再跑一次 hash 审计。

当前 portable manifests 的确切计数见 `dataset_manifests/emotiontalk/summary.json`；重新运行转换后应以新生成的 `summary.json` 和 `group_audit.jsonl` 为准。

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
| `student_spk_audio` | Base SFT/OPD student | 与 target 分离的同 speaker enrollment；VoiceDesign 忽略 |
| `teacher_ref_audio` | OPD teacher | 同 speaker、同 scene 的下一条音频 |
| `teacher_ref_codes_path` | OPD teacher | 缓存 ICL reference 的 `[T,16]` codec `.npy` |
| `teacher_ref_spk_emb_path` | OPD teacher | 缓存 ICL reference 的 speaker embedding `.npy` |
| `teacher_ref_text` | OPD teacher | teacher reference 的准确转写 |

`ref_audio/ref_text` 仅为兼容旧入口；新代码优先读取语义明确的字段。

### 从 GitHub manifest 在另一台服务器恢复

仓库已经包含当前实验使用的 SFT/OPD split 和 cycle 配对，不需要在每台服务器重新生成随机切分。下载 gated 音频后执行：

```bash
python -m qwen3opsd.portable_manifest materialize \
  --input-dir dataset_manifests/emotiontalk \
  --output-dir data/processed/emotiontalk \
  --audio-root data/raw/emotiontalk/extracted \
  --check-audio
```

如果目标目录已存在且确认需要替换对应 manifest，显式增加 `--overwrite`。本机重新导出 portable manifests 的命令是：

```bash
python -m qwen3opsd.portable_manifest export \
  --input-dir data/processed/emotiontalk \
  --output-dir dataset_manifests/emotiontalk \
  --audio-root data/raw/emotiontalk/extracted \
  --overwrite
```

`sft_train_with_codes.jsonl` 不上传：它约 90 MB，而且 codec 取决于目标服务器实际使用的 Qwen tokenizer/checkpoint。materialize 后按下一节运行 `scripts/prepare_sft.sh` 即可重建。

模型权重也不适合进入 GitHub：单个 `model.safetensors` 约 3.85 GB。要从现有 caption OPD 进度续跑，需通过 `rsync`、共享存储或私有模型仓库另行传输以下两个目录：

```text
checkpoints/emotiontalk_sft_caption/final/
checkpoints/emotiontalk_opd_caption/step_3000/
```

只传 GitHub 仓库时，可以在新服务器重新完成 SFT，再从头运行 OPD。

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

这里 `audio_codes` 的形状必须是 `[T, 16]`：外层长度 `T` 是 12 Hz codec 帧数，每一帧恰好包含 16 个 codebook token。示例只写了两帧用于展示结构，真实音频会有更多帧。`TRAIN_JSONL` 可以指向这个带 `audio_codes` 的文件，也可以使用下文带 `codes_path` 的缓存格式。

OPD 的 `INPUT_JSONL` 不需要预先生成 `audio_codes`，因为 trajectory 由 student 在线采样。每行推荐结构如下：

```jsonl
{"text":"今天很高兴见到你。","instruction":"女性声音清亮自然，语速稍快。","target_audio":"data/audio/target_001.wav","teacher_ref_audio":"data/audio/same_scene_002.wav","teacher_ref_text":"这是一条同场景参考语音的准确转写。","language":"Chinese"}
```

OPD 始终需要 `text`（或 `target_text`）。`base_icl` 模式需要 `teacher_ref_text`，并在 `teacher_ref_audio` 与缓存字段 `teacher_ref_codes_path + teacher_ref_spk_emb_path` 之间二选一；单独运行 `voice_design` teacher 模式不要求这些 ICL 字段。`instruction` 允许为空但 VoiceDesign 训练通常应提供，`language` 和仅用于泄漏检查/审计的 `target_audio` 可以省略。

### 4.2 已缓存 codes/spk embedding 的 JSONL

也支持如下已有数据，不需要把 `.npy` 展开写进 JSONL：

```jsonl
{"key":"小艺/sample_000001","text":"可以呀，你是发生什么事儿了？","codes_path":"/data/qwen3tts_codes/小艺/sample_000001.npy","spk_emb_path":"/data/qwen3tts_spk_emb/小艺/sample_000001.npy","language":"Auto","caption":"一位青年女性用清澈甜美的嗓音温柔地给予安慰。","caption_simplify_v1":"青年女性以温柔关怀的语气说话。"}
```

- `caption` 自动作为 VoiceDesign instruction；仅当它为空时才回退到 `caption_simplify_v1`。
- `codes_path` 可直接供 SFT 和 NLL qualification 读取，支持 `[T,16]`、`[16,T]` 或带单个 batch 维的形式；必须由与 student/Base 相同的 Qwen3-TTS 12 Hz tokenizer 生成。
- `spk_emb_path` 不会输入 VoiceDesign student。它只在作为另一条样本的 Base teacher ICL reference 时使用。
- OPD trajectory 仍由 student 在线生成，因此 target 自己的 `codes_path` 不参与 OPD loss。

Base ICL 不允许拿 target 自己当 reference。先按 speaker 配对；默认依次读取 `speaker_id`、`speaker`、`spk`，若都没有则使用 `key` 的第一段（上例为 `小艺`）：

```bash
INPUT_JSONL=/data/train_cached.jsonl \
OUTPUT_JSONL=/data/train_cached_opd.jsonl \
bash scripts/prepare_cached_opd.sh
```

转换器会保留原字段，并从同 speaker 的另一条不同文本样本增加：

```jsonl
{"key":"小艺/sample_000001","text":"可以呀，你是发生什么事儿了？","codes_path":"/data/qwen3tts_codes/小艺/sample_000001.npy","spk_emb_path":"/data/qwen3tts_spk_emb/小艺/sample_000001.npy","language":"Auto","caption":"一位青年女性用清澈甜美的嗓音温柔地给予安慰。","teacher_ref_key":"小艺/sample_000002","teacher_ref_codes_path":"/data/qwen3tts_codes/小艺/sample_000002.npy","teacher_ref_spk_emb_path":"/data/qwen3tts_spk_emb/小艺/sample_000002.npy","teacher_ref_text":"这是 reference 音频对应的准确文本。"}
```

这三个 `teacher_ref_*` 字段共同构造完整 Base ICL：`ref_code + ref_spk_embedding + ref_text`，内部固定为 `x_vector_only_mode=False`、`icl_mode=True`。若你的 speaker 字段另有名称，例如 `speaker_name`，设置 `SPEAKER_FIELD=speaker_name`。

## 5. Instruction SFT

先提取 target audio codes：

```bash
MODEL_PATH=/absolute/path/Qwen3-TTS-12Hz-1.7B-Base \
DEVICE=cuda:0 \
BATCH_SIZE=16 \
bash scripts/prepare_sft.sh
```

然后训练：

```bash
MODEL_PATH=/absolute/path/Qwen3-TTS-12Hz-1.7B-Base \
TRAIN_JSONL=data/processed/emotiontalk/sft_train_with_codes.jsonl \
OUTPUT_DIR=checkpoints/emotiontalk_sft_caption \
BATCH_SIZE=1 \
GRAD_ACCUM_STEPS=8 \
NUM_EPOCHS=3 \
LR=2e-6 \
bash scripts/train_sft.sh --overwrite
```

训练约束：

- `scripts/train_sft.sh` 要求 `tts_model_type=base`，每条样本读取自己的 `student_spk_audio`。
- instruction 使用官方独立 user prompt，不会拼进 assistant target text。
- speaker encoder 和 speech tokenizer 冻结，只训练 talker。
- VoiceDesign 对照实验使用 `MODEL_PATH=... bash scripts/train_vd_sft.sh`，不读取 enrollment。

## 6. Teacher Qualification 与 OPD

OPD 支持两个冻结 teacher 候选：

- `base_icl`：Base teacher 读取 target text 和音频或缓存形式的 teacher ICL reference。
- `voice_design`：VD teacher 只读取与 student 相同的 instruction/target text，不使用 ICL。

先给 OPD validation 子集补 target `audio_codes`：

```bash
MODEL_PATH=/absolute/path/Qwen3-TTS-12Hz-1.7B-VoiceDesign \
INPUT_JSONL=data/processed/emotiontalk/opd_val.jsonl \
OUTPUT_JSONL=data/processed/emotiontalk/opd_val_with_codes.jsonl \
bash scripts/prepare_sft.sh
```

然后顺序加载 student 和 Base-ICL teacher，在相同真实 target codes 上比较 codec-0、EOS、sub-codebook 和总 NLL：

```bash
STUDENT_MODEL_PATH=checkpoints/emotiontalk_vd_sft/final \
BASE_TEACHER_MODEL_PATH=/absolute/path/Qwen3-TTS-12Hz-1.7B-Base \
INPUT_JSONL=data/processed/emotiontalk/opd_val_with_codes.jsonl \
DEVICE=cuda:0 \
MAX_SAMPLES=500 \
bash scripts/qualify_teachers.sh
```

缓存格式可直接把 `INPUT_JSONL` 设为 `train_cached_opd.jsonl`，脚本会读取其中的 `codes_path`。默认只比较 student 与 Base；设置 `VD_TEACHER_MODEL_PATH` 才会加入第三个 VD teacher。

该工具一次只在设备上保留一个模型，输出：

```text
results/teacher_qualification/scores.jsonl
results/teacher_qualification/summary.json
```

定义 `margin = student_total_nll - teacher_total_nll`。`positive_rate` 越高，teacher 在真实 target 上胜过 student 的样本比例越高。若 `same_student_and_vd_teacher_path=true`，VD teacher 与 student 是同一路径，这个比较通常没有蒸馏价值。NLL qualification 只验证 token 建模能力，正式长跑前仍应补充 WER、风格/情绪匹配和音质生成评测。

再运行三方生成推理，实际比较可懂度、音质和 instruction/情绪匹配。输入使用原始 `opd_val.jsonl` 即可，不要求 `audio_codes`：

```bash
STUDENT_MODEL_PATH=checkpoints/emotiontalk_vd_sft/final \
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
base_icl:   Base teacher + teacher reference ICL + target text
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
OUTPUT_DIR=checkpoints/emotiontalk_opd_caption \
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

从已有的 `step_N` 中断点继续时，不要传 `--overwrite`。训练器会从目录名恢复全局步数，按相同 seed 重建 shuffle，并跳过已经完成的数据行。旧 checkpoint 没有保存 Adam 状态，因此第一次续跑会保留模型权重但重新初始化优化器动量。

```bash
STUDENT_MODEL_PATH=checkpoints/emotiontalk_sft_caption/final \
TEACHER_MODEL_PATH=checkpoints/emotiontalk_sft_caption/final \
INPUT_JSONL=data/processed/emotiontalk/opd_train.jsonl \
OUTPUT_DIR=checkpoints/emotiontalk_opd_caption \
DEVICE=cuda:0 \
TEACHER_DEVICE=cuda:1 \
NUM_EPOCHS=1 \
bash scripts/train_opd.sh --shuffle \
  --resume-from-checkpoint checkpoints/emotiontalk_opd_caption/step_3000
```

单张 80 GB GPU 也可以把 `DEVICE` 和 `TEACHER_DEVICE` 都设为 `cuda:0`；两张 GPU 会降低单卡显存压力。

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

1. student 按模型类型读取 Base x-vector enrollment 或 VoiceDesign instruction，采样 codec codes。
2. 所选 teacher 按模式读取 Base ICL 条件或 VD instruction/text 条件。
3. teacher 和 student 都 replay **同一条 student codes**。
4. 优化首 codebook KL、子 codebook KL 和小权重 student token CE。

## 7. 推理

推理入口自动识别 Base 或 VoiceDesign checkpoint；Base 需要 enrollment，VoiceDesign 可省略：

```bash
python -m qwen3opsd.infer \
  --model-path checkpoints/emotiontalk_opd_caption/final \
  --student-spk-audio /absolute/path/enrollment.wav \
  --instruction "女性声音清亮自然，语速稍快，语气轻松。" \
  --text "今天很高兴见到你。" \
  --output-wav outputs/demo.wav \
  --device cuda:0
```

训练和推理必须使用 Qwen 的独立 caption/instruct 通道：

```text
instruct_ids: <|im_start|>user\n<caption><|im_end|>\n
input_ids:    <|im_start|>assistant\n<transcript><|im_end|>\n<|im_start|>assistant\n
```

`caption_1` 写入数据行的 `instruction` 字段，训练时编码为 `instruct_ids`；`text` 只包含需要朗读的 transcript。严禁把两者拼成 `Instruction: ...\nText: ...` 后送入 `input_ids`，否则模型会把 caption 当作朗读内容。student 和 teacher 使用同一 caption，teacher 仍额外使用 `teacher_ref_audio/text` 作为 ICL 特权信息。

## 8. EmotionTalk 三模型评测

批量推理脚本统一使用 `sft_test.jsonl` 的 instruction、target text 和独立 enrollment，支持逐条写 manifest 和断点续跑。三个 run 必须使用相同 `SEED` 和 generation 参数：

```bash
# Base：caption 走独立 instruct 通道，作为未经微调的控制基线
MODEL_PATH=/opt/data/private/jsj/Qwen3-TTS-12Hz-1.7B-Base \
MODEL_NAME=base_instruction \
OUTPUT_DIR=outputs/emotiontalk_eval_caption/base_instruction \
DEVICE=cuda:0 \
bash scripts/eval_emotiontalk.sh

# SFT
MODEL_PATH=checkpoints/emotiontalk_sft_caption/final \
MODEL_NAME=sft_instruction \
OUTPUT_DIR=outputs/emotiontalk_eval_caption/sft_instruction \
DEVICE=cuda:0 \
bash scripts/eval_emotiontalk.sh

# OPD
MODEL_PATH=checkpoints/emotiontalk_opd_caption/final \
MODEL_NAME=opd_instruction \
OUTPUT_DIR=outputs/emotiontalk_eval_caption/opd_instruction \
DEVICE=cuda:0 \
bash scripts/eval_emotiontalk.sh
```

用 `--limit 3` 做 smoke test；再次执行不带 `--limit` 会跳过已经成功生成的样本并继续完整测试集。错误样本默认不会反复重试，可显式传 `--retry-errors`。

三个 run 有共同完成的样本后，生成盲听页面、盲化映射和 API judge JSONL：

```bash
python -m qwen3opsd.build_eval_report \
  --input-jsonl data/processed/emotiontalk/sft_test.jsonl \
  --run Base=outputs/emotiontalk_eval_caption/base_instruction/manifest.jsonl \
  --run SFT=outputs/emotiontalk_eval_caption/sft_instruction/manifest.jsonl \
  --run OPD=outputs/emotiontalk_eval_caption/opd_instruction/manifest.jsonl \
  --output-dir outputs/emotiontalk_eval_caption/report
```

打开 `outputs/emotiontalk_eval_caption/report/listen.html` 可以逐条听 enrollment、ground truth 和随机盲化后的三个系统。页面评分保存在浏览器 local storage，并可导出 JSON。`api_judge.jsonl` 保留未盲化的模型名和绝对音频路径，供 Gemini 或其他 Audio-LLM judge 使用。

如果需要原始语音克隆内容基线，再单独运行 Base 并设置 `CONDITIONING=text_only`；不要用它替代同协议的 `base_instruction`。

### InstructTTSEval 人工试听

官方仓库和数据集分别下载到 `/opt/data/private/jsj/InstructTTSEval` 与 `/opt/data/private/jsj/InstructTTSEval-dataset`。以下命令从中文 1000 条中确定性选择互不重叠的 APS/DSD/RP 各 20 条，解出 GT reference audio，并为 voice-clone 模型配置独立的男/女 enrollment：

```bash
python -m qwen3opsd.prepare_instructttseval \
  --parquet /opt/data/private/jsj/InstructTTSEval-dataset/zh.parquet \
  --language zh \
  --output-dir data/processed/instructttseval_zh_60 \
  --male-enrollment data/raw/emotiontalk/extracted/Audio/wav/G00003/G00003_11/G00003_11_13/G00003_11_13_001.wav \
  --female-enrollment data/raw/emotiontalk/extracted/Audio/wav/G00003/G00003_11/G00003_11_02/G00003_11_02_002.wav \
  --num-per-task 20 \
  --seed 20260716
```

输出的 `zh_eval.jsonl` 可直接作为 `scripts/eval_emotiontalk.sh` 的 `INPUT_JSONL`。三模型完成后继续用 `build_eval_report` 生成试听页；页面逐条展示 instruction、文本、独立 enrollment、官方 GT 和随机盲化的模型音频。GT 只用于试听对比，绝不作为模型输入。`--num-per-task 0` 会展开完整的 3000 个中文 task 实例。

## 9. 测试与短程检查

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

## 10. 常见问题

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
