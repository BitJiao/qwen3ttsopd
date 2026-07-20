from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW

from qwen3tts_opd.core import (
    conditioned_token_logits,
    ensure_qwen3_tts_repo_on_path,
    first_codebook_labels_with_eos,
    first_codebook_logits_with_eos,
    finish_time,
    format_eta,
    generate_student_codes,
    load_jsonl,
    load_tts,
    resolve_local_model_dir,
    save_checkpoint,
    token_ce,
    token_kl,
    torch_dtype,
)
from qwen3tts_opd.instruction_utils import get_target_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="On-policy distillation trainer for Qwen3-TTS.")
    parser.add_argument("--student_model_path", "--model_path", dest="student_model_path", required=True)
    parser.add_argument("--teacher_model_path", required=True)
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--output_dir", default="checkpoints/qwen3_tts_opd")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--teacher_device", default=None)
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--teacher_dtype", default=None, choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--attn_implementation", default="sdpa")
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--kl_temperature", type=float, default=1.0)
    parser.add_argument("--sub_kl_weight", type=float, default=0.3)
    parser.add_argument("--student_ce_weight", type=float, default=0.05)
    parser.add_argument("--save_freq", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--subtalker_temperature", type=float, default=0.9)
    parser.add_argument("--subtalker_top_k", type=int, default=50)
    parser.add_argument("--subtalker_top_p", type=float, default=1.0)
    parser.add_argument("--non_streaming_mode", action="store_true", default=True)
    parser.add_argument("--streaming_mode", dest="non_streaming_mode", action="store_false")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _device(requested: str) -> torch.device:
    if requested.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


def _validate_row(row: dict, *, row_number: int | None = None) -> None:
    location = f" {row_number}" if row_number is not None else ""
    try:
        get_target_text(row)
    except KeyError as exc:
        raise KeyError(f"OPD row{location} requires text (or target_text)") from exc
    teacher_audio = row.get("teacher_ref_audio", row.get("ref_audio"))
    teacher_text = row.get("teacher_ref_text", row.get("ref_text"))
    if not teacher_audio:
        raise KeyError(f"OPD row{location} requires teacher_ref_audio (or legacy ref_audio)")
    if not teacher_text:
        raise KeyError(f"OPD row{location} requires teacher_ref_text (or legacy ref_text) for teacher ICL")
    target_audio = row.get("target_audio", row.get("audio"))
    if target_audio and os.path.realpath(target_audio) == os.path.realpath(teacher_audio):
        raise ValueError(f"OPD row{location} has identical target_audio and teacher_ref_audio")


def main() -> None:
    ensure_qwen3_tts_repo_on_path()
    args = parse_args()
    torch.set_num_threads(max(1, int(os.environ.get("TORCH_NUM_THREADS", "1"))))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    data = load_jsonl(args.input_jsonl)
    if not data:
        raise ValueError(f"no rows loaded from {args.input_jsonl}")
    for row_number, row in enumerate(data, start=1):
        _validate_row(row, row_number=row_number)

    student_dir = resolve_local_model_dir(args.student_model_path)
    teacher_dir = resolve_local_model_dir(args.teacher_model_path)
    student_device = _device(args.device)
    teacher_device = _device(args.teacher_device or args.device)
    student_dtype = torch_dtype(args.dtype if student_device.type != "cpu" else "fp32")
    teacher_dtype = torch_dtype((args.teacher_dtype or args.dtype) if teacher_device.type != "cpu" else "fp32")

    student = load_tts(student_dir, student_dtype, args.attn_implementation, student_device)
    teacher = load_tts(teacher_dir, teacher_dtype, args.attn_implementation, teacher_device)
    if student.model.tts_model_type != "voice_design":
        raise ValueError(f"OPD student must be VoiceDesign, got {student.model.tts_model_type}")
    if teacher.model.tts_model_type != "base" or getattr(teacher.model, "speaker_encoder", None) is None:
        raise ValueError(f"OPD ICL teacher must be Base, got {teacher.model.tts_model_type}")
    if student.model.talker.config.num_code_groups != teacher.model.talker.config.num_code_groups:
        raise ValueError("student and teacher use different numbers of codec codebooks")
    if student.model.talker.config.vocab_size != teacher.model.talker.config.vocab_size:
        raise ValueError("student and teacher codec vocabularies are incompatible")
    if (
        student.model.config.talker_config.codec_eos_token_id
        != teacher.model.config.talker_config.codec_eos_token_id
    ):
        raise ValueError("student and teacher use different codec EOS tokens")
    if student.model.tokenizer_type != teacher.model.tokenizer_type:
        raise ValueError(
            f"student and teacher use different speech tokenizers: "
            f"{student.model.tokenizer_type} vs {teacher.model.tokenizer_type}"
        )

    for param in student.model.parameters():
        param.requires_grad_(False)
    for param in student.model.talker.parameters():
        param.requires_grad_(True)
    teacher.model.eval()
    for param in teacher.model.parameters():
        param.requires_grad_(False)

    output_root = Path(args.output_dir)
    if output_root.exists() and args.overwrite:
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    optimizer = AdamW(student.model.talker.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    planned_steps = len(data) * args.num_epochs
    if args.max_steps > 0:
        planned_steps = min(planned_steps, args.max_steps)

    global_step = 0
    begin = time.perf_counter()
    for epoch in range(args.num_epochs):
        if args.shuffle:
            random.shuffle(data)

        for row in data:
            student_codes = generate_student_codes(student, row, args)
            if student_codes.numel() == 0:
                continue

            with torch.no_grad():
                teacher_logits = conditioned_token_logits(
                    teacher,
                    row,
                    student_codes.to(teacher_device),
                    conditioning="teacher_icl",
                    non_streaming_mode=args.non_streaming_mode,
                )

            student.model.train()
            student_logits = conditioned_token_logits(
                student,
                row,
                student_codes.to(student_device),
                conditioning="voice_design",
                non_streaming_mode=args.non_streaming_mode,
            )

            student_first_logits = first_codebook_logits_with_eos(student_logits)
            teacher_first_logits = first_codebook_logits_with_eos(teacher_logits).to(student_device)
            first_kl = token_kl(
                student_first_logits,
                teacher_first_logits,
                args.kl_temperature,
            )
            sub_kl = token_kl(
                student_logits.sub_codebooks.reshape(-1, student_logits.sub_codebooks.shape[-1]),
                teacher_logits.sub_codebooks.to(student_device).reshape(-1, teacher_logits.sub_codebooks.shape[-1]),
                args.kl_temperature,
            )
            first_labels = first_codebook_labels_with_eos(student.model, student_codes, student_device)
            ce = token_ce(student_first_logits, first_labels)
            loss = first_kl + args.sub_kl_weight * sub_kl + args.student_ce_weight * ce

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(student.model.parameters(), args.max_grad_norm)
            optimizer.step()

            global_step += 1
            elapsed = time.perf_counter() - begin
            avg = elapsed / max(global_step, 1)
            remaining = max(planned_steps - global_step, 0)
            print(
                json.dumps(
                    {
                        "epoch": epoch,
                        "step": global_step,
                        "total_steps": planned_steps,
                        "tokens": int(student_codes.shape[0]),
                        "loss": float(loss.detach().cpu()),
                        "first_kl": float(first_kl.detach().cpu()),
                        "sub_kl": float(sub_kl.detach().cpu()),
                        "student_ce": float(ce.detach().cpu()),
                        "grad_norm": float(grad_norm.detach().cpu() if torch.is_tensor(grad_norm) else grad_norm),
                        "elapsed": format_eta(elapsed),
                        "eta": format_eta(remaining * avg),
                        "finish_at": finish_time(remaining * avg),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

            if args.save_freq > 0 and global_step % args.save_freq == 0:
                save_checkpoint(student, student_dir, str(output_root / f"step_{global_step}"), overwrite=True)
            if args.max_steps > 0 and global_step >= args.max_steps:
                save_checkpoint(student, student_dir, str(output_root / "final"), overwrite=True)
                return

    save_checkpoint(student, student_dir, str(output_root / "final"), overwrite=True)


if __name__ == "__main__":
    main()
