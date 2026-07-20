from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from torch.optim import AdamW
from torch.utils.data import DataLoader

from qwen3opsd.instruction_utils import with_formatted_text
from qwen3opsd.sft_dataset import InstructionSFTDataset, validate_sft_row
from qwen3tts_opd.alignment import next_token_codec_mask
from qwen3tts_opd.core import load_jsonl, resolve_local_model_dir, save_checkpoint


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
    parser.add_argument("--instruction-template", choices=["qwen_control", "plain", "bracket"], default="qwen_control")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _speaker_embedding(model, path: str, cache: dict[str, torch.Tensor]) -> torch.Tensor:
    if path not in cache:
        audio, _ = librosa.load(path, sr=model.speaker_encoder_sample_rate, mono=True)
        cache[path] = model.extract_speaker_embedding(audio.astype(np.float32), model.speaker_encoder_sample_rate).detach().cpu()
    return cache[path].to(device=model.device, dtype=model.dtype)


def main() -> None:
    from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel

    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rows = [with_formatted_text(row, template=args.instruction_template) for row in load_jsonl(args.train_jsonl)]
    if not rows:
        raise ValueError(f"no rows loaded from {args.train_jsonl}")
    for row_number, row in enumerate(rows, start=1):
        validate_sft_row(row, row_number=row_number)
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

    dataset = InstructionSFTDataset(rows, tts.processor, tts.model.config)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=dataset.collate_fn)
    optimizer = AdamW((parameter for parameter in tts.model.talker.parameters() if parameter.requires_grad), lr=args.lr, weight_decay=args.weight_decay)
    model, optimizer, dataloader = accelerator.prepare(tts.model, optimizer, dataloader)
    tts.model = model
    tts.device = accelerator.device
    raw_model = accelerator.unwrap_model(model)
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"output directory is not empty: {output_dir}; pass --overwrite")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    speaker_cache: dict[str, torch.Tensor] = {}
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
                input_ids = batch["input_ids"]
                codec_ids = batch["codec_ids"]
                text_embedding = raw_model.talker.text_projection(raw_model.talker.get_text_embeddings()(input_ids[:, :, 0]))
                text_embedding = text_embedding * batch["text_embedding_mask"]
                codec_embedding = raw_model.talker.get_input_embeddings()(input_ids[:, :, 1])
                codec_embedding = codec_embedding * batch["codec_embedding_mask"]
                speaker_embeddings = torch.stack(
                    [_speaker_embedding(raw_model, path, speaker_cache) for path in batch["student_spk_audio"]]
                )
                codec_embedding[:, 6, :] = speaker_embeddings
                input_embeddings = text_embedding + codec_embedding
                for index in range(1, raw_model.talker.config.num_code_groups):
                    sub_embedding = raw_model.talker.code_predictor.get_input_embeddings()[index - 1](codec_ids[:, :, index])
                    input_embeddings = input_embeddings + sub_embedding * batch["codec_mask"].unsqueeze(-1)
                outputs = raw_model.talker(
                    inputs_embeds=input_embeddings[:, :-1],
                    attention_mask=batch["attention_mask"][:, :-1],
                    output_hidden_states=True,
                )
                label_mask = batch["codec_0_labels"][:, 1:].ne(-100)
                first_loss = F.cross_entropy(outputs.logits[label_mask], batch["codec_0_labels"][:, 1:][label_mask])
                hidden = outputs.hidden_states[0][-1][next_token_codec_mask(batch["codec_mask"])]
                target_codes = codec_ids[batch["codec_mask"]]
                sub_logits, _ = raw_model.talker.forward_sub_talker_finetune(target_codes, hidden)
                sub_loss = F.cross_entropy(
                    sub_logits.reshape(-1, sub_logits.shape[-1]),
                    target_codes[:, 1:].reshape(-1),
                )
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
