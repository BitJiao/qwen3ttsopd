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


def get_target_audio(row: dict, *, row_number: int | None = None) -> str:
    audio = row.get("target_audio") or row.get("audio")
    if not audio:
        location = f" {row_number}" if row_number is not None else ""
        raise KeyError(f"SFT row{location} requires target_audio (or legacy audio)")
    return str(audio)


def load_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}: {exc.msg}") from exc
            if not isinstance(row, dict):
                raise TypeError(
                    f"each JSONL line must be an object, got {type(row).__name__} at {path}:{line_number}"
                )
            get_target_audio(row, row_number=line_number)
            rows.append(row)
    return rows


def main() -> None:
    from qwen_tts import Qwen3TTSTokenizer

    args = parse_args()
    rows = load_rows(args.input_jsonl)
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        raise ValueError(f"no rows loaded from {args.input_jsonl}")

    tokenizer_path = Path(args.model_path)
    if tokenizer_path.is_dir() and (tokenizer_path / "speech_tokenizer" / "config.json").is_file():
        tokenizer_path = tokenizer_path / "speech_tokenizer"
    tokenizer = Qwen3TTSTokenizer.from_pretrained(str(tokenizer_path), device_map=args.device)
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as output:
        for start in range(0, len(rows), args.batch_size):
            batch = rows[start : start + args.batch_size]
            encoded = tokenizer.encode([get_target_audio(row) for row in batch])
            if len(encoded.audio_codes) != len(batch):
                raise RuntimeError(
                    f"speech tokenizer returned {len(encoded.audio_codes)} code sequences for a batch of {len(batch)}"
                )
            for row, codes in zip(batch, encoded.audio_codes):
                row["audio_codes"] = codes.cpu().tolist()
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(json.dumps({"encoded": min(start + len(batch), len(rows)), "total": len(rows)}), flush=True)


if __name__ == "__main__":
    main()
