from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qwen3tts_opd.instruction_utils import get_instruction, get_target_text


@dataclass(frozen=True)
class CachedVoiceClonePromptItem:
    ref_code: Any
    ref_spk_embedding: Any
    x_vector_only_mode: bool
    icl_mode: bool
    ref_text: str


def _load_cached_tensor(path: str, *, kind: str):
    import numpy as np
    import torch

    resolved = Path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"{kind} path does not exist: {resolved}")
    value = np.asarray(np.load(resolved, allow_pickle=False))
    if kind == "teacher reference codes":
        if value.ndim == 3 and value.shape[0] == 1:
            value = value[0]
        if value.ndim == 2 and value.shape[1] != 16 and value.shape[0] == 16:
            value = value.T
        if value.ndim != 2 or value.shape[0] == 0 or value.shape[1] != 16:
            raise ValueError(f"{kind} must have shape [T, 16], got {value.shape} from {resolved}")
        return torch.as_tensor(value, dtype=torch.long)
    value = np.squeeze(value)
    if value.ndim != 1 or value.size == 0:
        raise ValueError(f"{kind} must be a non-empty vector, got {value.shape} from {resolved}")
    return torch.as_tensor(value, dtype=torch.float32)


def teacher_prompt_items(tts, sample: dict[str, Any]):
    ref_audio = sample.get("teacher_ref_audio", sample.get("ref_audio"))
    ref_text = sample.get("teacher_ref_text", sample.get("ref_text"))
    if not ref_text:
        raise KeyError("sample requires teacher_ref_text (or legacy ref_text)")

    ref_codes_path = sample.get("teacher_ref_codes_path", sample.get("ref_codes_path"))
    ref_spk_emb_path = sample.get("teacher_ref_spk_emb_path", sample.get("ref_spk_emb_path"))
    if ref_codes_path or ref_spk_emb_path:
        if not ref_codes_path:
            raise KeyError("cached teacher ICL requires teacher_ref_codes_path")
        if not ref_spk_emb_path:
            raise KeyError("cached teacher ICL requires teacher_ref_spk_emb_path")
        return [
            CachedVoiceClonePromptItem(
                ref_code=_load_cached_tensor(str(ref_codes_path), kind="teacher reference codes"),
                ref_spk_embedding=_load_cached_tensor(
                    str(ref_spk_emb_path), kind="teacher reference speaker embedding"
                ),
                x_vector_only_mode=False,
                icl_mode=True,
                ref_text=str(ref_text),
            )
        ]

    if not ref_audio:
        raise KeyError(
            "sample requires teacher_ref_audio, or cached teacher_ref_codes_path and teacher_ref_spk_emb_path"
        )
    return tts.create_voice_clone_prompt(
        ref_audio=[ref_audio], ref_text=[ref_text], x_vector_only_mode=[False]
    )


def teacher_icl_inputs(tts, sample: dict[str, Any]):
    prompt_items = teacher_prompt_items(tts, sample)
    prompt = tts._prompt_items_to_voice_clone_prompt(prompt_items)
    input_id = tts._tokenize_texts([tts._build_assistant_text(get_target_text(sample))])[0]

    resolved_ref_text = prompt_items[0].ref_text
    ref_id = None
    if resolved_ref_text is not None and resolved_ref_text != "":
        ref_id = tts._tokenize_texts([tts._build_ref_text(resolved_ref_text)])[0]
    return input_id, ref_id, prompt


def voice_design_inputs(tts, sample: dict[str, Any]):
    input_id = tts._tokenize_texts([tts._build_assistant_text(get_target_text(sample))])[0]
    instruction = get_instruction(sample)
    instruct_id = None
    if instruction:
        instruct_id = tts._tokenize_texts([tts._build_instruct_text(instruction)])[0]
    return input_id, instruct_id
