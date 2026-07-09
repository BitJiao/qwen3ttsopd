from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoConfig

from qwen3tts_opd.core import (
    build_training_batch,
    ensure_qwen3_tts_repo_on_path,
    finish_time,
    format_eta,
    load_jsonl,
    load_tts,
    qwen3_tts_nll,
    resolve_local_model_dir,
    save_checkpoint,
    torch_dtype,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline preference distillation trainer for Qwen3-TTS pairs.")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--pair_jsonl", required=True)
    parser.add_argument("--output_dir", default="checkpoints/qwen3_tts_opd")
    parser.add_argument("--ref_model_path", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--attn_implementation", default="sdpa")
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--sft_weight", type=float, default=0.2)
    parser.add_argument("--sub_talker_loss_coef", type=float, default=0.3)
    parser.add_argument("--save_freq", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _codes_tensor(row: dict[str, Any], key: str) -> torch.Tensor:
    return torch.tensor(row[key], dtype=torch.long)


def _build_pair_batches(row: dict[str, Any], processor, config, device: torch.device):
    base = {
        "text": row["text"],
        "ref_audio": row["ref_audio"],
        "ref_text": row.get("ref_text"),
        "language": row.get("language", "Auto"),
    }
    chosen = build_training_batch(base, _codes_tensor(row, "chosen_codes"), processor, config, device)
    rejected = build_training_batch(base, _codes_tensor(row, "rejected_codes"), processor, config, device)
    return chosen, rejected


def dpo_loss(policy_chosen_nll, policy_rejected_nll, ref_chosen_nll, ref_rejected_nll, beta: float):
    policy_logratio = -policy_chosen_nll + policy_rejected_nll
    ref_logratio = -ref_chosen_nll + ref_rejected_nll
    return -F.logsigmoid(beta * (policy_logratio - ref_logratio))


def main() -> None:
    ensure_qwen3_tts_repo_on_path()
    args = parse_args()
    torch.set_num_threads(max(1, int(os.environ.get("TORCH_NUM_THREADS", "1"))))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    local_model_dir = resolve_local_model_dir(args.model_path)
    ref_model_dir = resolve_local_model_dir(args.ref_model_path or args.model_path)
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    dtype = torch_dtype(args.dtype if device.type != "cpu" else "fp32")

    tts = load_tts(local_model_dir, dtype, args.attn_implementation, device)
    if getattr(tts.model, "speaker_encoder", None) is None:
        raise ValueError("Qwen3-TTS OPD requires a Base checkpoint with speaker_encoder.")
    ref_tts = load_tts(ref_model_dir, dtype, args.attn_implementation, device)
    ref_tts.model.eval()
    for param in ref_tts.model.parameters():
        param.requires_grad_(False)

    config = AutoConfig.from_pretrained(local_model_dir)
    data = load_jsonl(args.pair_jsonl)
    if not data:
        raise ValueError(f"no pairs loaded from {args.pair_jsonl}")

    output_root = Path(args.output_dir)
    if output_root.exists() and args.overwrite:
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    optimizer = AdamW(tts.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    planned_steps = len(data) * args.num_epochs
    if args.max_steps > 0:
        planned_steps = min(planned_steps, args.max_steps)

    global_step = 0
    begin = time.perf_counter()
    for epoch in range(args.num_epochs):
        if args.shuffle:
            random.shuffle(data)

        for row in data:
            tts.model.train()
            chosen_batch, rejected_batch = _build_pair_batches(row, tts.processor, config, device)

            with torch.no_grad():
                ref_chosen_nll, _, _ = qwen3_tts_nll(
                    ref_tts.model,
                    chosen_batch,
                    sub_talker_loss_coef=args.sub_talker_loss_coef,
                )
                ref_rejected_nll, _, _ = qwen3_tts_nll(
                    ref_tts.model,
                    rejected_batch,
                    sub_talker_loss_coef=args.sub_talker_loss_coef,
                )

            policy_chosen_nll, chosen_codec_loss, chosen_sub_loss = qwen3_tts_nll(
                tts.model,
                chosen_batch,
                sub_talker_loss_coef=args.sub_talker_loss_coef,
            )
            policy_rejected_nll, _, _ = qwen3_tts_nll(
                tts.model,
                rejected_batch,
                sub_talker_loss_coef=args.sub_talker_loss_coef,
            )

            pref_loss = dpo_loss(
                policy_chosen_nll,
                policy_rejected_nll,
                ref_chosen_nll,
                ref_rejected_nll,
                beta=args.beta,
            )
            loss = pref_loss + args.sft_weight * policy_chosen_nll

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(tts.model.parameters(), args.max_grad_norm)
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
                        "loss": float(loss.detach().cpu()),
                        "pref_loss": float(pref_loss.detach().cpu()),
                        "chosen_nll": float(policy_chosen_nll.detach().cpu()),
                        "rejected_nll": float(policy_rejected_nll.detach().cpu()),
                        "codec_0_loss": float(chosen_codec_loss.cpu()),
                        "sub_talker_loss": float(chosen_sub_loss.cpu()),
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
                save_checkpoint(tts, local_model_dir, str(output_root / f"step_{global_step}"), overwrite=True)
            if args.max_steps > 0 and global_step >= args.max_steps:
                save_checkpoint(tts, local_model_dir, str(output_root / "final"), overwrite=True)
                return

    save_checkpoint(tts, local_model_dir, str(output_root / "final"), overwrite=True)


if __name__ == "__main__":
    main()

