# EmotionTalk portable manifests

本目录保存本项目当前 EmotionTalk SFT/OPD 切分和配对结果，音频路径均相对于 EmotionTalk 解压根目录。它不包含原始音频、模型权重或预计算 codec。

数据来源：

- Dataset: <https://huggingface.co/datasets/BAAI/Emotiontalk>
- Metadata/code: <https://github.com/NKU-HLT/EmotionTalk>
- License: CC BY-NC-SA 4.0, non-commercial, gated access

文件：

- `sft_{train,val,test}.jsonl`: instruction SFT manifests
- `opd_{train,val,test}.jsonl`: ICL-teacher OPD manifests
- `group_audit.jsonl`: same-speaker, same-scene cycle audit
- `summary.json`: conversion summary
- `stats.json`: speaker split、情感分布和 OPD reference/target 同情感率

在新服务器上先按项目 README 下载并解压 EmotionTalk，然后执行：

```bash
python -m qwen3opsd.portable_manifest materialize \
  --input-dir dataset_manifests/emotiontalk \
  --output-dir data/processed/emotiontalk \
  --audio-root data/raw/emotiontalk/extracted \
  --check-audio
```

materialize 后的 JSONL 会使用目标服务器的绝对路径。SFT 还需要执行 `scripts/prepare_sft.sh` 生成本机模型对应的 `sft_train_with_codes.jsonl`；不要在不同 Qwen tokenizer/checkpoint 之间复用预计算 codec。

这些 manifest 是 EmotionTalk 的衍生数据库内容。使用或再分发时需遵守原始 CC BY-NC-SA 4.0 条款，包括署名、非商业和相同方式共享；manifest 的存在不授予 gated 原始音频访问权。
