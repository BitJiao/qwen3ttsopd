from __future__ import annotations

from typing import Any


def get_instruction(item: dict[str, Any]) -> str:
    for key in (
        "instruction",
        "caption",
        "caption_simplify_v1",
        "final_audio_caption",
        "emotion_instruction",
        "style_instruction",
        "instruct",
    ):
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
