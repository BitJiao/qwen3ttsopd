from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from torch.optim import AdamW
from torch.utils.data import DataLoader

from qwen3opsd.sft_dataset import InstructionSFTDataset
from qwen3tts_opd.core import conditioned_token_logits, load_jsonl, resolve_local_model_dir, save_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-speaker instruction SFT for Qwen3-TTS Base.")
    parser.add_argument("--init-model-path", required=True)
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--output-dir", default="checkpoints/emotiontalk_sft")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--lr", type=float, default=2e-6)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--sub-loss-weight", type=float, default=0.3)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--save-freq", type=int, default=500)
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
        raise ValueError("train_sft currently supports one process; use gradient accumulation on one GPU")
    model_dir = resolve_local_model_dir(args.init_model_path)
    load_dtype = torch.bfloat16 if args.mixed_precision == "bf16" else torch.float16 if args.mixed_precision == "fp16" else torch.float32
    tts = Qwen3TTSModel.from_pretrained(
        model_dir,
        dtype=load_dtype,
        attn_implementation=args.attn_implementation,
    )
    if tts.model.tts_model_type != "base" or tts.model.speaker_encoder is None:
        raise ValueError("instruction SFT requires a Qwen3-TTS Base checkpoint")
    for parameter in tts.model.parameters():
        parameter.requires_grad_(False)
    for parameter in tts.model.talker.parameters():
        parameter.requires_grad_(True)

    rows = load_jsonl(args.train_jsonl)
    if not rows:
        raise ValueError(f"no rows loaded from {args.train_jsonl}")
    dataset = InstructionSFTDataset(rows)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=dataset.collate_fn)
    optimizer = AdamW((parameter for parameter in tts.model.talker.parameters() if parameter.requires_grad), lr=args.lr, weight_decay=args.weight_decay)
    model, optimizer, dataloader = accelerator.prepare(tts.model, optimizer, dataloader)
    tts.model = model
    tts.device = accelerator.device
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"output directory is not empty: {output_dir}; pass --overwrite")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt_cache: dict[tuple[str, str | None, bool], object] = {}
    global_step = 0

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
                for sample, target_codes in zip(batch["samples"], batch["audio_codes"]):
                    target_codes = target_codes.to(accelerator.device)
                    logits = conditioned_token_logits(
                        tts,
                        sample,
                        target_codes,
                        x_vector_only_mode=True,
                        non_streaming_mode=True,
                        prompt_cache=prompt_cache,
                    )
                    first_losses.append(F.cross_entropy(logits.first_codebook, target_codes[:, 0]))
                    sub_losses.append(F.cross_entropy(
                        logits.sub_codebooks.reshape(-1, logits.sub_codebooks.shape[-1]),
                        target_codes[:, 1:].reshape(-1),
                    ))
                first_loss = torch.stack(first_losses).mean()
                sub_loss = torch.stack(sub_losses).mean()
                loss = first_loss + args.sub_loss_weight * sub_loss
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            if accelerator.sync_gradients:
                global_step += 1
                accelerator.print(json.dumps({"epoch": epoch, "step": global_step, "loss": float(loss.detach()), "first_loss": float(first_loss.detach()), "sub_loss": float(sub_loss.detach())}))
                if args.save_freq > 0 and global_step % args.save_freq == 0:
                    save(f"step_{global_step}")
                if args.max_steps > 0 and global_step >= args.max_steps:
                    save("final")
                    return
        save(f"epoch_{epoch}")
    save("final")


if __name__ == "__main__":
    main()
