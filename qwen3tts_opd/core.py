from __future__ import annotations

import json
import math
import os
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from safetensors.torch import save_file


def ensure_qwen3_tts_repo_on_path() -> None:
    env_repo = os.environ.get("QWEN3_TTS_REPO")
    if env_repo and env_repo not in sys.path:
        sys.path.insert(0, env_repo)


ensure_qwen3_tts_repo_on_path()

@dataclass
class TokenLogits:
    first_codebook: torch.Tensor
    sub_codebooks: torch.Tensor


def load_jsonl(path: str) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def torch_dtype(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    return torch.float32


def resolve_local_model_dir(model_path: str) -> str:
    if os.path.isdir(model_path):
        return model_path
    return snapshot_download(model_path)


def move_tts_to_device(tts, device: torch.device | str):
    device = torch.device(device)
    tts.model.to(device)
    tts.device = device

    speech_tokenizer = getattr(tts.model, "speech_tokenizer", None)
    if speech_tokenizer is not None and getattr(speech_tokenizer, "model", None) is not None:
        speech_tokenizer.model.to(device)
        speech_tokenizer.device = device

    return tts


def load_tts(local_model_dir: str, dtype: torch.dtype, attn_implementation: str, device: torch.device | str):
    from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel

    tts = Qwen3TTSModel.from_pretrained(
        local_model_dir,
        dtype=dtype,
        attn_implementation=attn_implementation,
    )
    return move_tts_to_device(tts, device)


def _prompt_for_sample(tts, sample: dict[str, Any], *, x_vector_only_mode: bool):
    prompt_items = tts.create_voice_clone_prompt(
        ref_audio=[sample["ref_audio"]],
        ref_text=[sample.get("ref_text")],
        x_vector_only_mode=[x_vector_only_mode],
    )
    prompt = tts._prompt_items_to_voice_clone_prompt(prompt_items)
    input_id = tts._tokenize_texts([tts._build_assistant_text(sample["text"])])[0]

    ref_id = None
    ref_text = prompt_items[0].ref_text
    if ref_text is not None and ref_text != "":
        ref_id = tts._tokenize_texts([tts._build_ref_text(ref_text)])[0]

    return input_id, ref_id, prompt


def _language_id(model, language: str | None):
    if language is None or language.lower() == "auto":
        return None
    language_norm = language.lower()
    if language_norm not in model.config.talker_config.codec_language_id:
        raise NotImplementedError(f"Language {language} not implemented")
    return model.config.talker_config.codec_language_id[language_norm]


def build_voice_clone_prefill(
    tts,
    sample: dict[str, Any],
    *,
    x_vector_only_mode: bool,
    non_streaming_mode: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build the same voice-clone prefill embeddings used by Qwen3-TTS generate()."""

    model = tts.model
    talker = model.talker
    device = next(model.parameters()).device
    input_id, ref_id, prompt = _prompt_for_sample(tts, sample, x_vector_only_mode=x_vector_only_mode)
    input_id = input_id.to(device)

    voice_clone_spk_embeds = model.generate_speaker_prompt(prompt)
    speaker_embed = None
    if prompt["x_vector_only_mode"][0] or prompt["icl_mode"][0]:
        speaker_embed = voice_clone_spk_embeds[0].to(device).to(next(model.parameters()).dtype)

    language_id = _language_id(model, sample.get("language", "Auto"))
    tts_bos_embed, tts_eos_embed, tts_pad_embed = talker.text_projection(
        talker.get_text_embeddings()(
            torch.tensor(
                [[model.config.tts_bos_token_id, model.config.tts_eos_token_id, model.config.tts_pad_token_id]],
                device=device,
                dtype=input_id.dtype,
            )
        )
    ).chunk(3, dim=1)

    if language_id is None:
        codec_prefill = [
            [
                model.config.talker_config.codec_nothink_id,
                model.config.talker_config.codec_think_bos_id,
                model.config.talker_config.codec_think_eos_id,
            ]
        ]
    else:
        codec_prefill = [
            [
                model.config.talker_config.codec_think_id,
                model.config.talker_config.codec_think_bos_id,
                language_id,
                model.config.talker_config.codec_think_eos_id,
            ]
        ]

    codec_prefill_embed = talker.get_input_embeddings()(torch.tensor(codec_prefill, device=device, dtype=input_id.dtype))
    codec_tail_embed = talker.get_input_embeddings()(
        torch.tensor(
            [[model.config.talker_config.codec_pad_id, model.config.talker_config.codec_bos_id]],
            device=device,
            dtype=input_id.dtype,
        )
    )
    if speaker_embed is None:
        codec_input_embedding = torch.cat([codec_prefill_embed, codec_tail_embed], dim=1)
    else:
        codec_input_embedding = torch.cat(
            [codec_prefill_embed, speaker_embed.view(1, 1, -1), codec_tail_embed],
            dim=1,
        )

    role_embed = talker.text_projection(talker.get_text_embeddings()(input_id[:, :3]))
    tag_embed = torch.cat(
        (
            tts_pad_embed.expand(-1, codec_input_embedding.shape[1] - 2, -1),
            tts_bos_embed,
        ),
        dim=1,
    ) + codec_input_embedding[:, :-1]
    prefill = torch.cat((role_embed, tag_embed), dim=1)

    ref_code = prompt["ref_code"][0] if prompt.get("ref_code") is not None else None
    if ref_code is not None and prompt["icl_mode"][0]:
        if ref_id is None:
            raise ValueError("ref_text is required for teacher ICL conditioning")
        icl_embed, trailing_text_hidden = model.generate_icl_prompt(
            text_id=input_id[:, 3:-5],
            ref_id=ref_id.to(device)[:, 3:-2],
            ref_code=ref_code.to(device),
            tts_pad_embed=tts_pad_embed,
            tts_eos_embed=tts_eos_embed,
            non_streaming_mode=non_streaming_mode,
        )
        return torch.cat([prefill, icl_embed], dim=1), trailing_text_hidden, tts_pad_embed

    prefill = torch.cat(
        [
            prefill,
            talker.text_projection(talker.get_text_embeddings()(input_id[:, 3:4])) + codec_input_embedding[:, -1:],
        ],
        dim=1,
    )
    if non_streaming_mode:
        prefill = prefill[:, :-1]
        text_embed = torch.cat(
            (
                talker.text_projection(talker.get_text_embeddings()(input_id[:, 3:-5])),
                tts_eos_embed,
            ),
            dim=1,
        )
        text_codec_pad = talker.get_input_embeddings()(
            torch.tensor(
                [[model.config.talker_config.codec_pad_id] * text_embed.shape[1]],
                device=device,
                dtype=input_id.dtype,
            )
        )
        codec_bos_embed = talker.get_input_embeddings()(
            torch.tensor(
                [[model.config.talker_config.codec_bos_id]],
                device=device,
                dtype=input_id.dtype,
            )
        )
        prefill = torch.cat([prefill, text_embed + text_codec_pad, tts_pad_embed + codec_bos_embed], dim=1)
        trailing_text_hidden = tts_pad_embed
    else:
        trailing_text_hidden = torch.cat(
            (
                talker.text_projection(talker.get_text_embeddings()(input_id[:, 4:-5])),
                tts_eos_embed,
            ),
            dim=1,
        )

    return prefill, trailing_text_hidden, tts_pad_embed


def _codec_frame_embeddings(model, codes: torch.Tensor) -> torch.Tensor:
    talker = model.talker
    pieces = [talker.get_input_embeddings()(codes[:, 0])]
    for idx in range(1, model.talker.config.num_code_groups):
        pieces.append(talker.code_predictor.get_input_embeddings()[idx - 1](codes[:, idx]))
    return torch.stack(pieces, dim=0).sum(dim=0)


def _trailing_steps(trailing_text_hidden: torch.Tensor, tts_pad_embed: torch.Tensor, steps: int) -> torch.Tensor:
    if trailing_text_hidden.shape[1] >= steps:
        return trailing_text_hidden[:, :steps, :]
    pad = tts_pad_embed.expand(-1, steps - trailing_text_hidden.shape[1], -1)
    return torch.cat([trailing_text_hidden, pad], dim=1)


def conditioned_token_logits(
    tts,
    sample: dict[str, Any],
    codes: torch.Tensor,
    *,
    x_vector_only_mode: bool,
    non_streaming_mode: bool,
) -> TokenLogits:
    model = tts.model
    device = next(model.parameters()).device
    codes = codes.to(device=device, dtype=torch.long)
    if codes.ndim != 2:
        raise ValueError(f"codes must be [T, Q], got shape {tuple(codes.shape)}")
    if codes.shape[1] != model.talker.config.num_code_groups:
        raise ValueError(f"codes have {codes.shape[1]} codebooks, expected {model.talker.config.num_code_groups}")

    prefill, trailing_text_hidden, tts_pad_embed = build_voice_clone_prefill(
        tts,
        sample,
        x_vector_only_mode=x_vector_only_mode,
        non_streaming_mode=non_streaming_mode,
    )
    frame_text = _trailing_steps(trailing_text_hidden, tts_pad_embed, codes.shape[0])
    frame_embeds = _codec_frame_embeddings(model, codes).unsqueeze(0) + frame_text

    eos_id = torch.tensor(
        [[model.config.talker_config.codec_eos_token_id]],
        device=device,
        dtype=torch.long,
    )
    eos_text = _trailing_steps(trailing_text_hidden, tts_pad_embed, codes.shape[0] + 1)[:, -1:, :]
    eos_embed = model.talker.get_input_embeddings()(eos_id) + eos_text

    inputs_embeds = torch.cat([prefill, frame_embeds, eos_embed], dim=1)
    attention_mask = torch.ones(inputs_embeds.shape[:2], device=inputs_embeds.device, dtype=torch.long)
    outputs = model.talker(
        inputs_embeds=inputs_embeds[:, :-1, :],
        attention_mask=attention_mask[:, :-1],
        output_hidden_states=True,
    )

    start = prefill.shape[1]
    positions = torch.arange(start, start + codes.shape[0], device=device)
    first_logits = outputs.logits[0, positions - 1, :]

    hidden_states = outputs.hidden_states[0][-1]
    talker_hidden_states = hidden_states[0, positions, :]
    sub_logits, _ = model.talker.forward_sub_talker_finetune(codes, talker_hidden_states)
    return TokenLogits(first_codebook=first_logits, sub_codebooks=sub_logits)


@torch.no_grad()
def generate_student_codes(tts, sample: dict[str, Any], args) -> torch.Tensor:
    was_training = tts.model.training
    tts.model.eval()
    try:
        input_ids, ref_id, prompt = _prompt_for_sample(tts, sample, x_vector_only_mode=True)
        codes, _ = tts.model.generate(
            input_ids=[input_ids.to(tts.device)],
            ref_ids=[ref_id.to(tts.device) if ref_id is not None else None],
            voice_clone_prompt=prompt,
            languages=[sample.get("language", "Auto")],
            non_streaming_mode=args.non_streaming_mode,
            do_sample=True,
            top_k=args.top_k,
            top_p=args.top_p,
            temperature=args.temperature,
            subtalker_dosample=True,
            subtalker_top_k=args.subtalker_top_k,
            subtalker_top_p=args.subtalker_top_p,
            subtalker_temperature=args.subtalker_temperature,
            max_new_tokens=args.max_new_tokens,
        )
    finally:
        if was_training:
            tts.model.train()
    if not codes:
        raise RuntimeError("student rollout returned no codes")
    return codes[0].detach()


def token_kl(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float) -> torch.Tensor:
    student_logp = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_p = F.softmax(teacher_logits.detach() / temperature, dim=-1)
    return F.kl_div(student_logp, teacher_p, reduction="batchmean") * (temperature**2)


def token_ce(student_logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(student_logits.reshape(-1, student_logits.shape[-1]), labels.reshape(-1))


def remove_model_files(output_dir: str) -> None:
    output = Path(output_dir)
    patterns = ["model*.safetensors", "model.safetensors.index.json", "pytorch_model*.bin", "pytorch_model.bin.index.json"]
    for pattern in patterns:
        for path in output.glob(pattern):
            path.unlink()


def save_checkpoint(tts, base_model_dir: str, output_dir: str, overwrite: bool = True) -> None:
    output = Path(output_dir)
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"{output_dir} already exists")
        shutil.rmtree(output)
    shutil.copytree(base_model_dir, output_dir)
    remove_model_files(output_dir)
    state_dict = {key: value.detach().cpu().contiguous() for key, value in tts.model.state_dict().items()}
    save_file(state_dict, os.path.join(output_dir, "model.safetensors"))
    tts.processor.save_pretrained(output_dir)


def format_eta(seconds: float) -> str:
    if not math.isfinite(seconds) or seconds < 0:
        return "unknown"
    seconds = int(seconds)
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def finish_time(remaining_seconds: float) -> str:
    return (datetime.now() + timedelta(seconds=remaining_seconds)).strftime("%Y-%m-%d %H:%M:%S")
