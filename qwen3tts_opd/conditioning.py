from __future__ import annotations

from typing import Any

from qwen3tts_opd.instruction_utils import get_instruction, get_target_text


def teacher_icl_inputs(tts, sample: dict[str, Any]):
    ref_audio = sample.get("teacher_ref_audio", sample.get("ref_audio"))
    ref_text = sample.get("teacher_ref_text", sample.get("ref_text"))
    if not ref_audio:
        raise KeyError("sample requires teacher_ref_audio (or legacy ref_audio)")
    if not ref_text:
        raise KeyError("sample requires teacher_ref_text (or legacy ref_text)")
    prompt_items = tts.create_voice_clone_prompt(
        ref_audio=[ref_audio],
        ref_text=[ref_text],
        x_vector_only_mode=[False],
    )
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
