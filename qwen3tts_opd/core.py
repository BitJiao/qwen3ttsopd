from __future__ import annotations

import importlib
import importlib.util
import json
import math
import os
import shutil
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
from safetensors.torch import save_file


def ensure_qwen3_tts_repo_on_path() -> None:
    env_repo = os.environ.get("QWEN3_TTS_REPO")
    if env_repo and env_repo not in sys.path:
        sys.path.insert(0, env_repo)

    qwen_spec = importlib.util.find_spec("qwen_tts")
    if qwen_spec is not None and qwen_spec.origin:
        repo = str(Path(qwen_spec.origin).resolve().parents[1])
        if repo not in sys.path:
            sys.path.insert(0, repo)


def load_jsonl(path: str) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def torch_dtype(name: str):
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    return torch.float32


def resolve_local_model_dir(model_path: str) -> str:
    if os.path.isdir(model_path):
        return model_path
    return snapshot_download(model_path)


def move_tts_to_device(tts: Qwen3TTSModel, device: torch.device | str) -> Qwen3TTSModel:
    device = torch.device(device)
    tts.model.to(device)
    tts.device = device

    speech_tokenizer = getattr(tts.model, "speech_tokenizer", None)
    if speech_tokenizer is not None and getattr(speech_tokenizer, "model", None) is not None:
        speech_tokenizer.model.to(device)
        speech_tokenizer.device = device

    return tts


def load_tts(local_model_dir: str, dtype: torch.dtype, attn_implementation: str, device: torch.device | str):
    tts = Qwen3TTSModel.from_pretrained(
        local_model_dir,
        dtype=dtype,
        attn_implementation=attn_implementation,
    )
    return move_tts_to_device(tts, device)


def import_reward_fn(spec: str | None) -> Callable[..., float]:
    if spec is None:
        from qwen3tts_opd.reward import wer_sim_reward

        return wer_sim_reward.compute_score

    if ":" not in spec:
        raise ValueError("--reward_fn must be module:function or /path/file.py:function")

    module_name, fn_name = spec.split(":", 1)
    if module_name.endswith(".py") or os.path.exists(module_name):
        module_path = Path(module_name).resolve()
        loaded = importlib.util.spec_from_file_location(module_path.stem, module_path)
        if loaded is None or loaded.loader is None:
            raise ValueError(f"Cannot import reward module from {module_path}")
        module = importlib.util.module_from_spec(loaded)
        loaded.loader.exec_module(module)
    else:
        module = importlib.import_module(module_name)

    fn = getattr(module, fn_name)
    if not callable(fn):
        raise TypeError(f"Reward target is not callable: {spec}")
    return fn


def call_rewards(
    reward_fn: Callable[..., float],
    sample: dict[str, Any],
    wavs: list[np.ndarray],
    sample_rate: int,
    codes_list: list[torch.Tensor],
) -> list[float]:
    module = importlib.import_module(reward_fn.__module__)
    batch_fn = getattr(module, "compute_scores", None)
    if callable(batch_fn):
        try:
            values = batch_fn(sample=sample, wavs=wavs, sample_rate=sample_rate, audio_codes_list=codes_list)
        except TypeError:
            values = batch_fn(sample, wavs, sample_rate, codes_list)
        values = [float(value) for value in values]
        if len(values) != len(wavs):
            raise ValueError(f"Batch reward returned {len(values)} values for {len(wavs)} wavs")
        return values

    values = []
    for codes, wav in zip(codes_list, wavs):
        try:
            value = reward_fn(sample=sample, wav=wav, sample_rate=sample_rate, audio_codes=codes)
        except TypeError:
            value = reward_fn(sample, wav, sample_rate, codes)
        values.append(float(value))
    return values


@torch.no_grad()
def generate_voice_clone_rollouts(tts, sample: dict[str, Any], group_size: int, args):
    text = sample["text"]
    language = sample.get("language", "Auto")
    ref_audio = sample["ref_audio"]
    ref_text = sample.get("ref_text")

    texts = [text] * group_size
    languages = [language] * group_size
    ref_audios = [ref_audio] * group_size
    ref_texts = [ref_text] * group_size
    xvec_modes = [args.x_vector_only_mode] * group_size

    input_ids = tts._tokenize_texts([tts._build_assistant_text(t) for t in texts])
    prompt_items = tts.create_voice_clone_prompt(
        ref_audio=ref_audios,
        ref_text=ref_texts,
        x_vector_only_mode=xvec_modes,
    )
    voice_clone_prompt = tts._prompt_items_to_voice_clone_prompt(prompt_items)

    ref_ids = []
    for item in prompt_items:
        if item.ref_text is None or item.ref_text == "":
            ref_ids.append(None)
        else:
            ref_ids.append(tts._tokenize_texts([tts._build_ref_text(item.ref_text)])[0])

    codes, _ = tts.model.generate(
        input_ids=input_ids,
        ref_ids=ref_ids,
        voice_clone_prompt=voice_clone_prompt,
        languages=languages,
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

    decode_codes = []
    for idx, generated in enumerate(codes):
        ref_code = voice_clone_prompt.get("ref_code", [None] * group_size)[idx]
        if ref_code is None:
            decode_codes.append(generated)
        else:
            decode_codes.append(torch.cat([ref_code.to(generated.device), generated], dim=0))

    wavs_all, sample_rate = tts.model.speech_tokenizer.decode([{"audio_codes": c} for c in decode_codes])

    wavs = []
    for idx, wav in enumerate(wavs_all):
        ref_code = voice_clone_prompt.get("ref_code", [None] * group_size)[idx]
        if ref_code is None:
            wavs.append(wav)
            continue
        ref_len = int(ref_code.shape[0])
        total_len = int(decode_codes[idx].shape[0])
        cut = int(ref_len / max(total_len, 1) * wav.shape[0])
        wavs.append(wav[cut:])

    return codes, wavs, sample_rate


def _load_tts_dataset_class():
    ensure_qwen3_tts_repo_on_path()
    from finetuning.dataset import TTSDataset

    return TTSDataset


def build_training_batch(sample: dict[str, Any], codes: torch.Tensor, processor, config, device: torch.device):
    item = dict(sample)
    item.setdefault("audio", item["ref_audio"])
    item["audio_codes"] = codes.detach().cpu().tolist()
    dataset_cls = _load_tts_dataset_class()
    dataset = dataset_cls([item], processor, config)
    batch = dataset.collate_fn([dataset[0]])
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def qwen3_tts_nll(model, batch: dict[str, torch.Tensor], sub_talker_loss_coef: float):
    input_ids = batch["input_ids"]
    codec_ids = batch["codec_ids"]
    ref_mels = batch["ref_mels"]
    text_embedding_mask = batch["text_embedding_mask"]
    codec_embedding_mask = batch["codec_embedding_mask"]
    attention_mask = batch["attention_mask"]
    codec_0_labels = batch["codec_0_labels"]
    codec_mask = batch["codec_mask"]

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    speaker_embedding = model.speaker_encoder(ref_mels.to(device).to(dtype)).detach()
    input_text_ids = input_ids[:, :, 0]
    input_codec_ids = input_ids[:, :, 1]

    input_text_embedding = model.talker.text_projection(model.talker.model.text_embedding(input_text_ids))
    input_text_embedding = input_text_embedding * text_embedding_mask
    input_codec_embedding = model.talker.model.codec_embedding(input_codec_ids) * codec_embedding_mask
    input_codec_embedding[:, 6, :] = speaker_embedding

    input_embeddings = input_text_embedding + input_codec_embedding
    for idx in range(1, model.talker.config.num_code_groups):
        codec_i_embedding = model.talker.code_predictor.get_input_embeddings()[idx - 1](codec_ids[:, :, idx])
        codec_i_embedding = codec_i_embedding * codec_mask.unsqueeze(-1)
        input_embeddings = input_embeddings + codec_i_embedding

    codec_loss_mask = codec_0_labels[:, 1:].ne(-100)
    outputs = model.talker(
        inputs_embeds=input_embeddings[:, :-1, :],
        attention_mask=attention_mask[:, :-1],
        output_hidden_states=True,
    )
    codec_0_loss = F.cross_entropy(
        outputs.logits[codec_loss_mask],
        codec_0_labels[:, 1:][codec_loss_mask],
    )
    hidden_states = outputs.hidden_states[0][-1]
    talker_hidden_states = hidden_states[codec_mask[:, :-1]]
    talker_codec_ids = codec_ids[codec_mask]
    sub_talker_logits, _ = model.talker.forward_sub_talker_finetune(talker_codec_ids, talker_hidden_states)
    sub_talker_labels = talker_codec_ids[:, 1:]
    sub_talker_loss = F.cross_entropy(
        sub_talker_logits.reshape(-1, sub_talker_logits.size(-1)),
        sub_talker_labels.reshape(-1),
        ignore_index=-100,
    )
    loss = codec_0_loss + sub_talker_loss_coef * sub_talker_loss
    return loss, codec_0_loss.detach(), sub_talker_loss.detach()


def remove_model_files(output_dir: str):
    output = Path(output_dir)
    patterns = ["model*.safetensors", "model.safetensors.index.json", "pytorch_model*.bin", "pytorch_model.bin.index.json"]
    for pattern in patterns:
        for path in output.glob(pattern):
            path.unlink()


def save_checkpoint(tts, base_model_dir: str, output_dir: str, overwrite: bool = True):
    output = Path(output_dir)
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"{output_dir} already exists")
        shutil.rmtree(output)
    shutil.copytree(base_model_dir, output_dir)
    remove_model_files(output_dir)
    state_dict = {
        key: value.detach().cpu().contiguous()
        for key, value in tts.model.state_dict().items()
    }
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

