# Qwen3-TTS Instruction SFT + ICL Teacher OPD

本仓库实现面向 Qwen3-TTS Base 的两阶段训练：

1. **Instruction SFT**：student 输入自然语言语音风格 instruction、目标文本和独立的说话人 enrollment 音频。
2. **On-Policy Distillation (OPD)**：student 用自己的条件采样 codec trajectory；冻结 teacher 在相同 trajectory 上额外读取同场景参考音频和转写（ICL），逐 token 蒸馏给 student。

```text
student: instruction + target text + student_spk_audio (x-vector only)
teacher: instruction + target text + teacher_ref_audio/text (ICL)
action:  student 自己采样的 16-codebook codec trajectory
loss:    first-codebook KL + sub-codebook KL + small student CE
```

这里没有 ASR reward、chosen/rejected 数据、DPO loss 或基于 caption 相似度的配对筛选。

## 当前验证状态

截至 2026-08-03：

| 项目 | 结果 |
|---|---|
| EmotionTalk 数据 | gated 音频已下载，19,250 条 metadata 已全量转换 |
| 全量转换审计 | 1,452 个有效循环组 |
| 单元测试 | 22/22 通过 |
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
  prepare_codes.py     提取 Qwen3-TTS 12 Hz target codec
  sft_dataset.py       多说话人 instruction SFT dataset/collator
  train_sft.py         Base-compatible SFT trainer
  train_opd.py         OPD 入口
  infer.py             SFT/OPD checkpoint 推理
qwen3tts_opd/
  core.py              token replay、teacher/student logits、KL、保存
  train_opd.py         OPD 训练实现（保留旧包兼容）
scripts/
  setup_env.sh
  download_emotiontalk.sh
  convert_emotiontalk.sh
  prepare_sft.sh
  train_sft.sh
  train_opd.sh
tests/
```

原始数据、checkpoint、WAV、预计算 codec 和 Hugging Face token 均由 `.gitignore` 排除，不会推送到 GitHub。仓库中的 `dataset_manifests/emotiontalk/` 保存可迁移的相对路径清单；完整数据盘点见 [`docs/EMOTION_DATA_INVENTORY.md`](docs/EMOTION_DATA_INVENTORY.md)。

## 1. 环境

要求：

- Linux、Python 3.10+
- NVIDIA GPU；完整训练建议 40 GB 以上显存
- Qwen3-TTS 官方源码 checkout
- Qwen3-TTS-12Hz-1.7B-Base 或 0.6B-Base 权重
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

每个 split 内为每位 speaker 单独保留一条 enrollment，且该音频不再作为 target 或 teacher reference。这样 train 不会读取 validation/test enrollment。

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

仅 metadata 的全量审计结果（保留 enrollment 后）：

```json
{
  "source_rows": 19250,
  "enrollment_rows": 21,
  "sft_train": 15400,
  "sft_val": 1904,
  "sft_test": 1925,
  "opd_train": 15134,
  "opd_val": 1904,
  "opd_test": 1891,
  "valid_groups": 1452,
  "skipped_groups": {"same_text_pair": 20, "single_utterance": 4}
}
```

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
| `student_spk_audio` | SFT + OPD student | 独立 enrollment，仅提取 x-vector |
| `teacher_ref_audio` | OPD teacher | 同 speaker、同 scene 的下一条音频 |
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

与官方单说话人 SFT 的关键差别：

- 每条样本读取自己的 `student_spk_audio`，支持 EmotionTalk 多说话人。
- enrollment 自动重采样到 speaker encoder 需要的 24 kHz。
- speaker encoder 和 speech tokenizer 冻结，只训练 talker。
- checkpoint 保持 `tts_model_type=base`，仍支持 voice-clone/ICL；不会改成单一 `custom_voice` speaker。

## 6. ICL Teacher OPD

推荐 student 从 SFT checkpoint 开始。teacher 可以使用同一 SFT checkpoint 的冻结副本，使两边权重相同、只有 privileged conditioning 不同；也可以指定原始或更强的兼容 Base checkpoint。

```bash
STUDENT_MODEL_PATH=checkpoints/emotiontalk_sft_caption/final \
TEACHER_MODEL_PATH=checkpoints/emotiontalk_sft_caption/final \
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

每一步严格执行：

1. student 读取 instruction、target text、`student_spk_audio` 的 x-vector，采样 codec codes。
2. teacher 读取同样的 instruction/target text，加上 `teacher_ref_audio/text` ICL。
3. teacher 和 student 都 replay **同一条 student codes**。
4. 优化首 codebook KL、子 codebook KL 和小权重 student token CE。

## 7. 推理

SFT 和 OPD 输出仍是 Base checkpoint，推理时继续提供 enrollment：

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

**`Only support 24kHz audio`**

旧版官方 SFT dataset 直接 assert 24 kHz。本仓库的 enrollment loader 会自动重采样；target audio 由 Qwen tokenizer 正常化。

**OPD 显存不足**

把 teacher 放到另一张 GPU；减小 `MAX_NEW_TOKENS`；先使用 0.6B Base 做流程验证。当前实现是一条样本一次 on-policy rollout，不使用数据 batch。

**某些组被跳过**

查看 `group_audit.jsonl` 的 `reason`。不要为了增加数量关闭 self-reference、同音频或同文本检查，否则 teacher 条件可能直接泄漏 target。

## 许可证

本仓库代码遵循仓库自身许可证。Qwen3-TTS 权重与源码遵循其官方许可证。EmotionTalk 数据遵循 **CC BY-NC-SA 4.0**；使用者负责完成授权、署名、非商业限制和 ShareAlike 要求。
