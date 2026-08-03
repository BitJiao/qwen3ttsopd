# 情感与风格语音数据盘点

统计时间：2026-08-03。这里区分“当前 OPD 主线可直接使用的数据”和“已收集但需要重新审视配对语义的数据”。原始音频不随 Git 仓库分发；许可证、gated access 和署名要求仍由每台训练服务器上的使用者承担。

## 1. 当前主线：EmotionTalk

| 项目 | 统计 |
|---|---:|
| 官方 metadata | 19,250 条、19 位演员、7 种情感 |
| SFT train / val / test | 15,400 / 1,904 / 1,925 |
| OPD train / val / test | 15,134 / 1,904 / 1,891 |
| enrollment | 21 条，按 split 和 speaker 独立保留 |
| 有效 `(speaker, scene)` OPD 循环组 | 1,452 |
| 跳过组 | 同文本 pair 20，单 utterance 4 |
| 许可证 | CC BY-NC-SA 4.0，非商用，Hugging Face gated |

训练集情感分布：

| 情感 | SFT 条数 | OPD target 条数 | OPD reference 同情感 |
|---|---:|---:|---:|
| neutral | 7,563 | 7,418 | 4,891 (65.93%) |
| angry | 3,057 | 3,029 | 2,232 (73.69%) |
| happy | 1,713 | 1,695 | 762 (44.96%) |
| surprised | 1,087 | 1,053 | 260 (24.69%) |
| sad | 898 | 866 | 556 (64.20%) |
| disgusted | 612 | 605 | 105 (17.36%) |
| fearful | 470 | 468 | 163 (34.83%) |
| 合计 | 15,400 | 15,134 | 8,969 (59.26%) |

speaker split：

| split | speaker | 说明 |
|---|---|---|
| train | 05, 06, 07, 08, 09, 10, 11, 12, 14, 16, 17, 18, 19 | 13 人，七种情感均覆盖 |
| val | 01, 02, 13, 15 | 4 人，七种情感均覆盖 |
| test | 02, 13, 20, 21 | 4 人，七种情感均覆盖；与 train 无 speaker 重叠 |

这不是单说话人数据，但当前方法不要求先做单说话人版本：student 每条样本使用独立的同 speaker enrollment，teacher reference 严格来自同 speaker、同 scene，测试 speaker 与训练 speaker 分离。speaker identity 因此是显式条件，而不是让模型把“情感”和“人”混在一个固定说话人里。需要注意的是，当前 OPD 配对按场景相邻关系建立，并不强制 reference 与 target 同情感；只有 59.26% 的训练 pair 同情感。这是下一轮方法实验中最值得做的配对消融。

## 2. 已转换的兼容数据

本机位置：`/opt/data/private/jsj/datasets/processed/emotiontalk_style/`。这些统计来自各目录的 `summary.json`。

| 数据集 | 原始去重语音 | SFT train/val/test | 严格 OPD train/val/test | 许可证 | 当前判断 |
|---|---:|---:|---:|---|---|
| ExpressiveSpeech | 24,233 | 6,965 / 963 / 797 | 2,746 / 414 / 274 | CC BY-NC-SA 4.0 | 严格同 speaker 可用量小；高覆盖 OPD 大量跨 speaker，不宜直接并入主实验 |
| DailyTalk | 23,773 | 19,473 / 2,069 / 2,225 | 19,467 / 2,069 / 2,225 | CC BY-SA 4.0 | 当前镜像缺 emotion/dialog-act 标签，适合作为对话风格补充，不适合作为情感主监督 |
| Expresso | 11,615 | 10,393 / 624 / 586 | 10,380 / 624 / 584 | CC BY-NC 4.0 | 主要是 speaking style，不是统一七情感标签 |
| CREMA-D | 7,442 | 5,494 / 885 / 972 | 4,474 / 720 / 792 | ODbL 1.0 / DbCL 1.0 | actor/emotion/intensity 清楚，适合做显式情感专家或平衡实验 |
| Combined | 67,063 | 42,325 / 4,541 / 4,580 | 37,067 / 3,827 / 3,875 | 继承全部限制 | 只能在统一标签、配对协议和许可说明后使用 |

ExpressiveSpeech 的 high-coverage train view 有 19,240 条 OPD，但其中 16,064 条 teacher reference 是跨 speaker，11,887 条 enrollment 使用 target-as-x-vector fallback。它适合做弱监督/跨说话人 teacher 的单独消融，不能与 EmotionTalk 的严格同 speaker ICL 结果混报。

已知未完成项：ESD 和 MEAD 缺数据授权/音频；独立 IEMOCAP 未转换（ExpressiveSpeech 内含衍生子集）；独立 M3ED 只有 annotation（ExpressiveSpeech 内含衍生子集）。

## 3. 已收集的评测资产

| 资产 | 本机规模/状态 | 用途 |
|---|---|---|
| InstructTTSEval dataset | 约 1.2 GB；中英文 parquet 已下载 | instruction-following，已有中文 60 条试听子集脚本 |
| VStyle parquet/toolkit | parquet 约 775 MB | 风格可控评测候选 |
| S2S-Arena emotion | 24 个文件，约 6.8 MB | 小规模情感理解/生成检查 |
| StreamingBench Emotion Recognition | 压缩包约 4.3 GB | 情感识别评测，不是 OPD 训练数据 |

## 4. 建议的跨服务器实验顺序

1. 先用 GitHub 中的 EmotionTalk portable manifests 复现严格多说话人 SFT + OPD 主线。
2. 做 OPD pair 消融：当前同场景相邻、同情感优先匹配、不同情感 reference 三组；保持 target、speaker split 和 rollout seed 一致。
3. 用 CREMA-D 做 emotion/intensity 专家或数据平衡预实验。
4. Expresso 只作为 style 辅助任务；DailyTalk 只作为对话自然度辅助任务。
5. ExpressiveSpeech 严格 view 和 high-coverage view 分开报告，不把跨 speaker teacher 当成同 speaker privileged teacher。

GitHub 中的 `dataset_manifests/emotiontalk/` 是可迁移清单，不含音频和预计算 codec。其他数据当前只上传统计，不上传衍生 manifest，避免在配对语义尚未定稿时让另一台服务器误跑不一致实验。
