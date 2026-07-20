from __future__ import annotations

import argparse
import gc
import json
import random
import re
from pathlib import Path
from typing import Any

from qwen3opsd.instruction_utils import get_instruction, get_target_text


CANDIDATE_SPECS = {
    "student": ("student_model_path", "voice_design"),
    "base_icl": ("base_teacher_model_path", "base"),
    "vd_teacher": ("vd_teacher_model_path", "voice_design"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate comparable audio from the VoiceDesign student, Base-ICL teacher, and VD teacher."
    )
    parser.add_argument("--student_model_path", required=True)
    parser.add_argument("--base_teacher_model_path", required=True)
    parser.add_argument(
        "--vd_teacher_model_path",
        default=None,
        help="Optional frozen VoiceDesign teacher. Omit it for student-vs-Base comparison.",
    )
    parser.add_argument("--input_jsonl", required=True, help="OPD JSONL; audio_codes are not required.")
    parser.add_argument("--output_dir", default="results/inference_comparison")
    parser.add_argument("--device", default="cuda:0", help="Models are loaded sequentially on this device.")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--attn_implementation", default="sdpa")
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--do_sample", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--repetition_penalty", type=float, default=1.05)
    parser.add_argument("--subtalker_dosample", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--subtalker_top_k", type=int, default=50)
    parser.add_argument("--subtalker_top_p", type=float, default=1.0)
    parser.add_argument("--subtalker_temperature", type=float, default=0.9)
    parser.add_argument("--non_streaming_mode", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def candidate_specs(args: argparse.Namespace) -> dict[str, tuple[str, str]]:
    names = ["student", "base_icl"]
    if args.vd_teacher_model_path:
        names.append("vd_teacher")
    return {name: CANDIDATE_SPECS[name] for name in names}


def _device(requested: str):
    import torch

    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device {requested} was requested, but CUDA is unavailable")
    return torch.device(requested)


def _safe_sample_name(sample_id: Any, index: int) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(sample_id)).strip("._") or "sample"
    return f"{index:05d}_{slug[:96]}"


def _set_seed(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generation_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.do_sample,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "temperature": args.temperature,
        "repetition_penalty": args.repetition_penalty,
        "subtalker_dosample": args.subtalker_dosample,
        "subtalker_top_k": args.subtalker_top_k,
        "subtalker_top_p": args.subtalker_top_p,
        "subtalker_temperature": args.subtalker_temperature,
    }


def generate_candidate(
    tts,
    candidate: str,
    row: dict[str, Any],
    *,
    non_streaming_mode: bool,
    gen_kwargs: dict[str, Any],
):
    common = {
        "text": get_target_text(row),
        "language": row.get("language", "Auto"),
        "non_streaming_mode": non_streaming_mode,
        **gen_kwargs,
    }
    if candidate == "base_icl":
        return tts.generate_voice_clone(
            **common,
            ref_audio=row.get("teacher_ref_audio", row.get("ref_audio")),
            ref_text=row.get("teacher_ref_text", row.get("ref_text")),
            x_vector_only_mode=False,
        )
    if candidate in {"student", "vd_teacher"}:
        return tts.generate_voice_design(**common, instruct=get_instruction(row))
    raise ValueError(f"unknown candidate: {candidate}")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _audio_metadata(path: Path, output_dir: Path) -> dict[str, Any]:
    import soundfile as sf

    info = sf.info(path)
    return {
        "wav": path.relative_to(output_dir).as_posix(),
        "sample_rate": info.samplerate,
        "samples": info.frames,
        "duration_seconds": info.duration,
    }


def _run_config(args: argparse.Namespace, model_paths: dict[str, str], gen_kwargs: dict[str, Any]) -> dict[str, Any]:
    return {
        "input_jsonl": str(Path(args.input_jsonl).resolve()),
        "model_paths": model_paths,
        "max_samples": args.max_samples,
        "seed": args.seed,
        "dtype": args.dtype,
        "attn_implementation": args.attn_implementation,
        "non_streaming_mode": args.non_streaming_mode,
        "generation": gen_kwargs,
    }


def _prepare_output(output_dir: Path, config: dict[str, Any], overwrite: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "run_config.json"
    if config_path.exists() and not overwrite:
        with config_path.open(encoding="utf-8") as handle:
            previous = json.load(handle)
        if previous != config:
            raise ValueError(
                f"{config_path} belongs to a different comparison run; use another output_dir or --overwrite"
            )
    _write_json(config_path, config)


def main() -> None:
    args = parse_args()

    import soundfile as sf
    import torch

    from qwen3tts_opd.core import (
        ensure_qwen3_tts_repo_on_path,
        load_jsonl,
        load_tts,
        resolve_local_model_dir,
        torch_dtype,
    )
    from qwen3tts_opd.teacher_modes import validate_opd_row

    ensure_qwen3_tts_repo_on_path()
    rows = load_jsonl(args.input_jsonl)
    if args.max_samples > 0:
        rows = rows[: args.max_samples]
    if not rows:
        raise ValueError(f"no rows loaded from {args.input_jsonl}")
    for row_number, row in enumerate(rows, start=1):
        get_target_text(row)
        validate_opd_row(row, "base_icl", row_number=row_number)

    device = _device(args.device)
    dtype = torch_dtype(args.dtype if device.type != "cpu" else "fp32")
    gen_kwargs = generation_kwargs(args)
    candidates = candidate_specs(args)
    model_paths = {name: str(getattr(args, fields[0])) for name, fields in candidates.items()}
    config = _run_config(args, model_paths, gen_kwargs)
    output_dir = Path(args.output_dir)
    _prepare_output(output_dir, config, args.overwrite)

    records: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        sample_id = row.get("sample_id", index)
        records.append(
            {
                "index": index,
                "sample_id": sample_id,
                "text": get_target_text(row),
                "instruction": get_instruction(row),
                "language": row.get("language", "Auto"),
                "target_audio": row.get("target_audio", row.get("audio")),
                "teacher_ref_audio": row.get("teacher_ref_audio", row.get("ref_audio")),
                "teacher_ref_text": row.get("teacher_ref_text", row.get("ref_text")),
                "outputs": {},
                "_name": _safe_sample_name(sample_id, index),
            }
        )

    signatures: dict[str, dict[str, Any]] = {}
    for candidate, (_, expected_type) in candidates.items():
        local_model_dir = resolve_local_model_dir(model_paths[candidate])
        tts = load_tts(local_model_dir, dtype, args.attn_implementation, device)
        actual_type = tts.model.tts_model_type
        if actual_type != expected_type:
            raise ValueError(f"{candidate} must be {expected_type}, got {actual_type}")
        tts.model.eval()
        signatures[candidate] = {
            "tts_model_type": actual_type,
            "tokenizer_type": tts.model.tokenizer_type,
        }

        for index, (row, record) in enumerate(zip(rows, records)):
            sample_seed = args.seed + index
            wav_path = output_dir / "audio" / candidate / f"{record['_name']}.wav"
            if args.overwrite or not wav_path.exists():
                _set_seed(sample_seed)
                wavs, sample_rate = generate_candidate(
                    tts,
                    candidate,
                    row,
                    non_streaming_mode=args.non_streaming_mode,
                    gen_kwargs=gen_kwargs,
                )
                if len(wavs) != 1:
                    raise RuntimeError(f"{candidate} returned {len(wavs)} waveforms for one sample")
                wav_path.parent.mkdir(parents=True, exist_ok=True)
                sf.write(wav_path, wavs[0], sample_rate)
            record["outputs"][candidate] = {"seed": sample_seed, **_audio_metadata(wav_path, output_dir)}
            print(json.dumps({"candidate": candidate, "generated": index + 1, "total": len(rows)}), flush=True)

        del tts
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    for record in records:
        del record["_name"]
    _write_jsonl(output_dir / "manifest.jsonl", records)
    summary = {
        "samples": len(records),
        "model_paths": model_paths,
        "signatures": signatures,
        "manifest": "manifest.jsonl",
        "generation": gen_kwargs,
        "seed_policy": "candidate sample i uses seed + i",
    }
    _write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
