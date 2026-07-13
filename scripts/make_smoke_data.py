from __future__ import annotations

import argparse
import csv
from pathlib import Path

import librosa
import soundfile as sf


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a four-utterance EmotionTalk-shaped smoke dataset.")
    parser.add_argument("--source-wav", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--segment-seconds", type=float, default=0.7)
    args = parser.parse_args()
    audio, _ = librosa.load(args.source_wav, sr=24000, mono=True)
    segment_frames = int(args.segment_seconds * 24000)
    if len(audio) < segment_frames * 4:
        raise ValueError("source wav is too short for four non-overlapping smoke segments")
    keys = [
        "G00002/G00002_01/G00002_01_14/G00002_01_14_001",
        "G00002/G00002_02/G00002_02_14/G00002_02_14_001",
        "G00002/G00002_02/G00002_02_14/G00002_02_14_002",
        "G00002/G00002_02/G00002_02_14/G00002_02_14_003",
    ]
    texts = ["这是说话人注册音频。", "今天天气很好。", "我们一起出发吧。", "记得带上你的雨伞。"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for index, key in enumerate(keys):
        path = args.output_dir / "audio" / f"{key}.wav"
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(path, audio[index * segment_frames : (index + 1) * segment_frames], 24000)
    with (args.output_dir / "transcription.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["name", "emotion", "chinese"])
        writer.writeheader()
        for key, text in zip(keys, texts):
            writer.writerow({"name": key, "emotion": "neutral", "chinese": text})
    with (args.output_dir / "audio.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["file_name", "emotion", "content"])
        writer.writeheader()
        for index, key in enumerate(keys):
            writer.writerow(
                {
                    "file_name": f"{key}.wav",
                    "emotion": "neutral",
                    "content": repr({"caption_1": f"女性声音自然清晰，语速平稳，样例风格编号 {index}。"}),
                }
            )


if __name__ == "__main__":
    main()
