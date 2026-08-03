from __future__ import annotations

import argparse
import json
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from torch.optim import AdamW
from torch.utils.data import DataLoader

from qwen3opsd.sft_dataset import VoiceDesignSFTDataset
from qwen3tts_opd.core import (
    finish_time,
    format_eta,
    load_jsonl,
    resolve_local_model_dir,
    save_checkpoint,
    voice_design_token_logits,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Caption SFT for Qwen3-TTS VoiceDesign (text+caption, no reference audio)."
    )
    parser.add_argument("--init-model-path", required=True)
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--output-dir", default="checkpoints/emotiontalk_vd_sft")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--num-epochs", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--lr", type=float, default=2e-6)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--sub-loss-weight", type=float, default=0.3)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--save-freq", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--mixed-precision", choices=["no", "fp16", "bf16"], default="bf16")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel

    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
    )
    if accelerator.num_processes != 1:
        raise ValueError("train_vd_sft supports one process; use accumulation on one GPU")
    model_dir = resolve_local_model_dir(args.init_model_path)
    load_dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "no": torch.float32,
    }[args.mixed_precision]
    tts = Qwen3TTSModel.from_pretrained(
        model_dir,
        dtype=load_dtype,
        attn_implementation=args.attn_implementation,
    )
    if tts.model.tts_model_type != "voice_design":
        raise ValueError("VD SFT requires a Qwen3-TTS VoiceDesign checkpoint")
    for parameter in tts.model.parameters():
        parameter.requires_grad_(False)
    for parameter in tts.model.talker.parameters():
        parameter.requires_grad_(True)

    rows = load_jsonl(args.train_jsonl)
    if not rows:
        raise ValueError(f"no rows loaded from {args.train_jsonl}")
    dataset = VoiceDesignSFTDataset(rows)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=dataset.collate_fn,
    )
    optimizer = AdamW(
        (parameter for parameter in tts.model.talker.parameters() if parameter.requires_grad),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    model, optimizer, dataloader = accelerator.prepare(tts.model, optimizer, dataloader)
    tts.model = model
    tts.device = accelerator.device

    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        if not args.overwrite:
            raise FileExistsError(
                f"output directory is not empty: {output_dir}; pass --overwrite"
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if accelerator.is_main_process:
        (output_dir / "training_config.json").write_text(
            json.dumps(
                {
                    **vars(args),
                    "conditioning": "target_text + caption",
                    "reference_audio_used": False,
                    "train_rows": len(rows),
                },
                ensure_ascii=False,
                indent=2,
                default=str,
            )
            + "\n",
            encoding="utf-8",
        )

    global_step = 0
    total_steps = args.num_epochs * max(
        1, (len(dataloader) + args.gradient_accumulation_steps - 1)
        // args.gradient_accumulation_steps,
    )
    if args.max_steps > 0:
        total_steps = min(total_steps, args.max_steps)
    started = time.monotonic()

    def save(name: str) -> None:
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            original = tts.model
            tts.model = accelerator.unwrap_model(model)
            save_checkpoint(tts, model_dir, str(output_dir / name), overwrite=True)
            tts.model = original

    model.train()
    for epoch in range(args.num_epochs):
        for batch in dataloader:
            with accelerator.accumulate(model):
                first_losses = []
                sub_losses = []
                eos_losses = []
                for sample, target_codes in zip(batch["samples"], batch["audio_codes"]):
                    target_codes = target_codes.to(accelerator.device)
                    logits = voice_design_token_logits(
                        tts,
                        sample,
                        target_codes,
                        non_streaming_mode=True,
                    )
                    if logits.codec_0_logits is None or logits.codec_0_labels is None:
                        raise RuntimeError("VoiceDesign replay did not return masked codec labels")
                    first_losses.append(
                        F.cross_entropy(logits.codec_0_logits, logits.codec_0_labels)
                    )
                    eos_losses.append(
                        F.cross_entropy(
                            logits.codec_0_logits[-1:], logits.codec_0_labels[-1:]
                        )
                    )
                    sub_losses.append(
                        F.cross_entropy(
                            logits.sub_codebooks.reshape(
                                -1, logits.sub_codebooks.shape[-1]
                            ),
                            target_codes[:, 1:].reshape(-1),
                        )
                    )
                first_loss = torch.stack(first_losses).mean()
                sub_loss = torch.stack(sub_losses).mean()
                eos_loss = torch.stack(eos_losses).mean()
                loss = first_loss + args.sub_loss_weight * sub_loss
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            if not accelerator.sync_gradients:
                continue
            global_step += 1
            if global_step == 1 or global_step % args.log_every == 0:
                elapsed = time.monotonic() - started
                remaining = elapsed / global_step * max(total_steps - global_step, 0)
                accelerator.print(
                    json.dumps(
                        {
                            "epoch": epoch,
                            "step": global_step,
                            "total_steps": total_steps,
                            "loss": round(float(loss.detach()), 6),
                            "first_loss": round(float(first_loss.detach()), 6),
                            "sub_loss": round(float(sub_loss.detach()), 6),
                            "eos_loss": round(float(eos_loss.detach()), 6),
                            "eta": format_eta(remaining),
                            "finish_at": finish_time(remaining),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
            if args.save_freq > 0 and global_step % args.save_freq == 0:
                save(f"step_{global_step}")
            if args.max_steps > 0 and global_step >= args.max_steps:
                save("final")
                return
        save(f"epoch_{epoch + 1}")
    save("final")


if __name__ == "__main__":
    main()
