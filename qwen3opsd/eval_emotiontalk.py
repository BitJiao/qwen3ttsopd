from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Any, Iterable

from qwen3opsd.instruction_utils import get_instruction, get_target_text


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
    return rows


def conditioned_text(row: dict[str, Any], conditioning: str) -> str:
    if conditioning in {"text_only", "instruction"}:
        return get_target_text(row).strip()
    raise ValueError(f"unknown conditioning mode: {conditioning}")


def stable_sample_seed(base_seed: int, sample_id: str) -> int:
    digest = hashlib.sha256(sample_id.encode("utf-8")).digest()
    return (base_seed + int.from_bytes(digest[:4], "big")) % (2**31)


def safe_filename(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    if not sanitized:
        raise ValueError(f"sample id cannot be converted to a filename: {value!r}")
    return sanitized


def latest_manifest_rows(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                print(f"[warning] ignoring incomplete manifest line {path}:{line_number}")
                continue
            sample_id = str(row.get("sample_id", ""))
            if sample_id:
                rows[sample_id] = row
    return rows


def append_jsonl(handle, row: dict[str, Any]) -> None:
    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def select_rows(rows: list[dict[str, Any]], offset: int, limit: int) -> list[dict[str, Any]]:
    if offset < 0:
        raise ValueError("offset must be non-negative")
    selected = rows[offset:]
    return selected[:limit] if limit > 0 else selected


def validate_rows(rows: Iterable[dict[str, Any]]) -> None:
    required = {"sample_id", "text", "instruction", "student_spk_audio", "target_audio"}
    seen: set[str] = set()
    for index, row in enumerate(rows):
        missing = sorted(required - row.keys())
        if missing:
            raise ValueError(f"row {index} is missing required fields: {missing}")
        sample_id = str(row["sample_id"])
        if sample_id in seen:
            raise ValueError(f"duplicate sample_id: {sample_id}")
        seen.add(sample_id)


def write_run_config(path: Path, config: dict[str, Any]) -> None:
    critical = (
        "model_path",
        "model_name",
        "input_jsonl",
        "conditioning",
        "dtype",
        "attn_implementation",
        "max_new_tokens",
        "seed",
    )
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        mismatches = [key for key in critical if existing.get(key) != config.get(key)]
        if mismatches:
            raise ValueError(
                f"output directory belongs to a different run; mismatched fields: {mismatches}"
            )
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def set_generation_seed(seed: int, torch_module, numpy_module) -> None:
    random.seed(seed)
    numpy_module.random.seed(seed)
    torch_module.manual_seed(seed)
    if torch_module.cuda.is_available():
        torch_module.cuda.manual_seed_all(seed)


def build_result_row(
    source: dict[str, Any],
    *,
    model_name: str,
    model_path: str,
    conditioning: str,
    generated_audio: Path,
    seed: int,
) -> dict[str, Any]:
    return {
        "sample_id": str(source["sample_id"]),
        "model_name": model_name,
        "model_path": model_path,
        "conditioning": conditioning,
        "seed": seed,
        "text": str(source["text"]),
        "instruction": str(source.get("instruction", "")),
        "conditioned_text": conditioned_text(source, conditioning),
        "benchmark": str(source.get("benchmark", "EmotionTalk")),
        "source_id": str(source.get("source_id", source["sample_id"])),
        "task": str(source.get("task", "")),
        "gender": str(source.get("gender", "")),
        "language": str(source.get("language", "Chinese")),
        "speaker_id": str(source.get("speaker_id", "")),
        "scene_id": str(source.get("scene_id", "")),
        "emotion": str(source.get("emotion", "")),
        "student_spk_audio": str(Path(source["student_spk_audio"]).resolve()),
        "target_audio": str(Path(source["target_audio"]).resolve()),
        "generated_audio": str(generated_audio.resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch EmotionTalk inference for Base-compatible Qwen3-TTS checkpoints."
    )
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--conditioning", choices=["instruction", "text_only"], default="instruction")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0, help="0 evaluates all remaining rows.")
    parser.add_argument("--retry-errors", action="store_true")
    parser.add_argument("--overwrite-audio", action="store_true")
    parser.add_argument("--log-every", type=int, default=10)
    args = parser.parse_args()

    if args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be positive")
    if args.log_every <= 0:
        parser.error("--log-every must be positive")

    model_path = args.model_path.expanduser().resolve()
    input_jsonl = args.input_jsonl.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    audio_dir = output_dir / "audio"
    manifest_path = output_dir / "manifest.jsonl"
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)

    rows = read_jsonl(input_jsonl)
    validate_rows(rows)
    rows = select_rows(rows, args.offset, args.limit)
    if not rows:
        raise ValueError("no evaluation rows selected")

    config = {
        "model_path": str(model_path),
        "model_name": args.model_name,
        "input_jsonl": str(input_jsonl),
        "conditioning": args.conditioning,
        "device": args.device,
        "dtype": args.dtype,
        "attn_implementation": args.attn_implementation,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
    }
    write_run_config(output_dir / "run_config.json", config)

    completed = latest_manifest_rows(manifest_path)
    pending: list[dict[str, Any]] = []
    for row in rows:
        previous = completed.get(str(row["sample_id"]))
        previous_audio = Path(previous["generated_audio"]) if previous and previous.get("generated_audio") else None
        if (
            previous
            and previous.get("status") == "ok"
            and previous_audio is not None
            and previous_audio.is_file()
            and previous_audio.stat().st_size > 0
            and not args.overwrite_audio
        ):
            continue
        if previous and previous.get("status") == "error" and not args.retry_errors:
            continue
        pending.append(row)

    print(
        json.dumps(
            {
                "selected": len(rows),
                "already_complete": len(rows) - len(pending),
                "pending": len(pending),
                "output_dir": str(output_dir),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    if not pending:
        return

    import numpy as np
    import soundfile as sf
    import torch
    from qwen_tts import Qwen3TTSModel
    from qwen3tts_opd.core import generate_instructed_voice_clone

    dtype = getattr(torch, args.dtype)
    model = Qwen3TTSModel.from_pretrained(
        str(model_path),
        device_map=args.device,
        dtype=dtype,
        attn_implementation=args.attn_implementation,
    )
    prompt_cache: dict[str, Any] = {}
    successful = 0
    failed = 0
    started = time.monotonic()

    with manifest_path.open("a", encoding="utf-8") as manifest:
        for index, source in enumerate(pending, 1):
            sample_id = str(source["sample_id"])
            output_wav = audio_dir / f"{safe_filename(sample_id)}.wav"
            sample_seed = stable_sample_seed(args.seed, sample_id)
            result = build_result_row(
                source,
                model_name=args.model_name,
                model_path=str(model_path),
                conditioning=args.conditioning,
                generated_audio=output_wav,
                seed=sample_seed,
            )
            sample_started = time.monotonic()
            try:
                enrollment = result["student_spk_audio"]
                if enrollment not in prompt_cache:
                    prompt_items = model.create_voice_clone_prompt(
                        ref_audio=enrollment,
                        ref_text=None,
                        x_vector_only_mode=True,
                    )
                    prompt_cache[enrollment] = model._prompt_items_to_voice_clone_prompt(prompt_items)
                set_generation_seed(sample_seed, torch, np)
                wavs, sample_rate = generate_instructed_voice_clone(
                    model,
                    text=result["conditioned_text"],
                    instruction=get_instruction(source) if args.conditioning == "instruction" else "",
                    language=result["language"],
                    voice_clone_prompt=prompt_cache[enrollment],
                    non_streaming_mode=True,
                    max_new_tokens=args.max_new_tokens,
                )
                temporary_wav = output_wav.with_suffix(".tmp.wav")
                sf.write(temporary_wav, wavs[0], sample_rate)
                temporary_wav.replace(output_wav)
                result.update(
                    {
                        "status": "ok",
                        "error": "",
                        "sample_rate": int(sample_rate),
                        "audio_seconds": float(len(wavs[0]) / sample_rate),
                        "generation_seconds": time.monotonic() - sample_started,
                    }
                )
                successful += 1
            except Exception as exc:
                result.update(
                    {
                        "status": "error",
                        "error": f"{type(exc).__name__}: {exc}",
                        "sample_rate": None,
                        "audio_seconds": None,
                        "generation_seconds": time.monotonic() - sample_started,
                    }
                )
                failed += 1
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            append_jsonl(manifest, result)

            if index == 1 or index % args.log_every == 0 or index == len(pending):
                elapsed = time.monotonic() - started
                print(
                    json.dumps(
                        {
                            "processed": index,
                            "pending_total": len(pending),
                            "successful": successful,
                            "failed": failed,
                            "last_sample_id": sample_id,
                            "elapsed_seconds": round(elapsed, 1),
                            "eta_seconds": round(elapsed / index * (len(pending) - index), 1),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )


if __name__ == "__main__":
    main()
