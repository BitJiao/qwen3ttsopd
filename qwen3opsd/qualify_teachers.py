from __future__ import annotations

import argparse
import gc
import json
from collections import Counter
from pathlib import Path
from statistics import mean

import torch

from qwen3opsd.data_contract import validate_sft_row
from qwen3opsd.instruction_utils import get_target_text
from qwen3tts_opd.core import (
    conditioned_token_logits,
    ensure_qwen3_tts_repo_on_path,
    first_codebook_labels_with_eos,
    first_codebook_logits_with_eos,
    load_jsonl,
    load_tts,
    resolve_local_model_dir,
    token_ce,
    torch_dtype,
)
from qwen3tts_opd.teacher_modes import validate_opd_row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare VoiceDesign student, Base-ICL teacher, and VD teacher NLL.")
    parser.add_argument("--student_model_path", required=True)
    parser.add_argument("--base_teacher_model_path", required=True)
    parser.add_argument("--vd_teacher_model_path", required=True)
    parser.add_argument("--input_jsonl", required=True, help="OPD JSONL with target audio_codes [T, 16].")
    parser.add_argument("--output_jsonl", default="results/teacher_qualification/scores.jsonl")
    parser.add_argument("--summary_json", default="results/teacher_qualification/summary.json")
    parser.add_argument("--device", default="cuda:0", help="Models are scored sequentially on this device.")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--attn_implementation", default="sdpa")
    parser.add_argument("--sub_loss_weight", type=float, default=0.3)
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--non_streaming_mode", action="store_true", default=True)
    parser.add_argument("--streaming_mode", dest="non_streaming_mode", action="store_false")
    return parser.parse_args()


def _device(requested: str) -> torch.device:
    if requested.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


def _model_signature(model) -> dict[str, object]:
    return {
        "tokenizer_type": model.tokenizer_type,
        "num_code_groups": int(model.talker.config.num_code_groups),
        "vocab_size": int(model.talker.config.vocab_size),
        "eos_token_id": int(model.config.talker_config.codec_eos_token_id),
    }


@torch.inference_mode()
def score_candidate(
    *,
    name: str,
    model_path: str,
    expected_model_type: str,
    conditioning: str,
    rows: list[dict],
    device: torch.device,
    dtype: torch.dtype,
    attn_implementation: str,
    non_streaming_mode: bool,
    sub_loss_weight: float,
) -> tuple[list[dict[str, float]], dict[str, object]]:
    local_model_dir = resolve_local_model_dir(model_path)
    tts = load_tts(local_model_dir, dtype, attn_implementation, device)
    if tts.model.tts_model_type != expected_model_type:
        raise ValueError(f"{name} must be {expected_model_type}, got {tts.model.tts_model_type}")
    tts.model.eval()
    for parameter in tts.model.parameters():
        parameter.requires_grad_(False)

    signature = _model_signature(tts.model)
    scores: list[dict[str, float]] = []
    for index, row in enumerate(rows, start=1):
        codes = torch.tensor(row["audio_codes"], dtype=torch.long, device=device)
        logits = conditioned_token_logits(
            tts,
            row,
            codes,
            conditioning=conditioning,
            non_streaming_mode=non_streaming_mode,
        )
        first_logits = first_codebook_logits_with_eos(logits)
        first_labels = first_codebook_labels_with_eos(tts.model, codes, first_logits.device)
        codec0_nll = token_ce(logits.first_codebook, codes[:, 0])
        eos_nll = token_ce(logits.eos, first_labels[-1:])
        first_nll = token_ce(first_logits, first_labels)
        sub_nll = token_ce(logits.sub_codebooks, codes[:, 1:])
        total_nll = first_nll + sub_loss_weight * sub_nll
        scores.append(
            {
                "codec0_nll": float(codec0_nll.cpu()),
                "eos_nll": float(eos_nll.cpu()),
                "first_nll": float(first_nll.cpu()),
                "sub_nll": float(sub_nll.cpu()),
                "total_nll": float(total_nll.cpu()),
            }
        )
        print(json.dumps({"candidate": name, "scored": index, "total": len(rows)}), flush=True)

    del tts
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return scores, signature


def _check_signatures(signatures: dict[str, dict[str, object]]) -> None:
    reference_name = "student"
    reference = signatures[reference_name]
    for name, signature in signatures.items():
        if signature != reference:
            raise ValueError(f"model signature mismatch: {reference_name}={reference}, {name}={signature}")


def build_summary(records: list[dict], model_paths: dict[str, str], sub_loss_weight: float) -> dict:
    summary: dict[str, object] = {
        "samples": len(records),
        "sub_loss_weight": sub_loss_weight,
        "model_paths": model_paths,
        "same_student_and_vd_teacher_path": model_paths["student"] == model_paths["vd_teacher"],
    }
    winners = Counter(record["winner"] for record in records)
    summary["winner_counts"] = dict(sorted(winners.items()))
    for candidate in ("student", "base_icl", "vd_teacher"):
        summary[candidate] = {
            key: mean(record[candidate][key] for record in records)
            for key in ("codec0_nll", "eos_nll", "first_nll", "sub_nll", "total_nll")
        }
    for candidate in ("base_icl", "vd_teacher"):
        margins = [record[f"{candidate}_margin"] for record in records]
        summary[f"{candidate}_margin"] = {
            "mean": mean(margins),
            "positive_rate": sum(value > 0 for value in margins) / len(margins),
        }
    return summary


def main() -> None:
    ensure_qwen3_tts_repo_on_path()
    args = parse_args()
    rows = load_jsonl(args.input_jsonl)
    if args.max_samples > 0:
        rows = rows[: args.max_samples]
    if not rows:
        raise ValueError(f"no rows loaded from {args.input_jsonl}")
    for row_number, row in enumerate(rows, start=1):
        validate_sft_row(row, row_number=row_number)
        validate_opd_row(row, "base_icl", row_number=row_number)

    device = _device(args.device)
    dtype = torch_dtype(args.dtype if device.type != "cpu" else "fp32")
    candidates = {
        "student": (args.student_model_path, "voice_design", "voice_design"),
        "base_icl": (args.base_teacher_model_path, "base", "teacher_icl"),
        "vd_teacher": (args.vd_teacher_model_path, "voice_design", "voice_design"),
    }
    all_scores: dict[str, list[dict[str, float]]] = {}
    signatures: dict[str, dict[str, object]] = {}
    for name, (model_path, model_type, conditioning) in candidates.items():
        all_scores[name], signatures[name] = score_candidate(
            name=name,
            model_path=model_path,
            expected_model_type=model_type,
            conditioning=conditioning,
            rows=rows,
            device=device,
            dtype=dtype,
            attn_implementation=args.attn_implementation,
            non_streaming_mode=args.non_streaming_mode,
            sub_loss_weight=args.sub_loss_weight,
        )
    _check_signatures(signatures)

    records = []
    for index, row in enumerate(rows):
        scores = {name: all_scores[name][index] for name in candidates}
        base_margin = scores["student"]["total_nll"] - scores["base_icl"]["total_nll"]
        vd_margin = scores["student"]["total_nll"] - scores["vd_teacher"]["total_nll"]
        winner = min(scores, key=lambda name: scores[name]["total_nll"])
        records.append(
            {
                "index": index,
                "sample_id": row.get("sample_id", index),
                "text": get_target_text(row),
                **scores,
                "base_icl_margin": base_margin,
                "vd_teacher_margin": vd_margin,
                "winner": winner,
            }
        )

    output_jsonl = Path(args.output_jsonl)
    summary_json = Path(args.summary_json)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    model_paths = {name: values[0] for name, values in candidates.items()}
    summary = build_summary(records, model_paths, args.sub_loss_weight)
    summary["signatures"] = signatures
    with summary_json.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
