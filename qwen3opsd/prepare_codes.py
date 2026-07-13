from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Add Qwen3-TTS 12 Hz target audio codes to SFT JSONL.")
    parser.add_argument("--model-path", required=True, help="Base checkpoint containing speech_tokenizer/.")
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def main() -> None:
    from qwen_tts import Qwen3TTSTokenizer

    args = parse_args()
    tokenizer_path = Path(args.model_path)
    if tokenizer_path.is_dir() and (tokenizer_path / "speech_tokenizer" / "config.json").is_file():
        tokenizer_path = tokenizer_path / "speech_tokenizer"
    tokenizer = Qwen3TTSTokenizer.from_pretrained(str(tokenizer_path), device_map=args.device)
    with args.input_jsonl.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if args.limit is not None:
        rows = rows[: args.limit]
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as output:
        for start in range(0, len(rows), args.batch_size):
            batch = rows[start : start + args.batch_size]
            encoded = tokenizer.encode([row.get("target_audio", row["audio"]) for row in batch])
            for row, codes in zip(batch, encoded.audio_codes):
                row["audio_codes"] = codes.cpu().tolist()
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(json.dumps({"encoded": min(start + len(batch), len(rows)), "total": len(rows)}), flush=True)


if __name__ == "__main__":
    main()
