from __future__ import annotations

from typing import Any


def get_instruction(item: dict[str, Any]) -> str:
    for key in ("instruction", "emotion_instruction", "style_instruction", "instruct"):
        value = item.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def get_target_text(item: dict[str, Any]) -> str:
    for key in ("text", "target_text", "target"):
        value = item.get(key)
        if value is not None:
            return str(value)
    raise KeyError("row must contain text or target_text")


def format_instruction_text(item: dict[str, Any], *, template: str = "qwen_control") -> str:
    text = get_target_text(item).strip()
    instruction = get_instruction(item)
    if not instruction:
        return text

    if template == "plain":
        return f"{instruction}\n{text}"
    if template == "bracket":
        return f"[Instruction] {instruction}\n[Text] {text}"
    if template == "qwen_control":
        return f"Instruction: {instruction}\nText: {text}"
    raise ValueError(f"unknown instruction template: {template}")


def with_formatted_text(item: dict[str, Any], *, template: str = "qwen_control") -> dict[str, Any]:
    out = dict(item)
    out.setdefault("raw_text", get_target_text(item))
    out["text"] = format_instruction_text(item, template=template)
    return out

