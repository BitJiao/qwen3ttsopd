from __future__ import annotations

from typing import Any

from qwen3opsd.instruction_utils import get_target_text


def validate_sft_row(item: dict[str, Any], *, row_number: int | None = None) -> None:
    location = f" {row_number}" if row_number is not None else ""
    try:
        get_target_text(item)
    except KeyError as exc:
        raise KeyError(f"SFT row{location} requires text (or target_text)") from exc
    if "audio_codes" not in item:
        raise KeyError(f"SFT row{location} requires audio_codes; run prepare_sft.sh first")
    audio_codes = item["audio_codes"]
    if not isinstance(audio_codes, list) or not audio_codes:
        raise ValueError(f"SFT row{location} audio_codes must be a non-empty [T, 16] list")
    for frame_index, frame in enumerate(audio_codes):
        if not isinstance(frame, list) or len(frame) != 16:
            width = len(frame) if isinstance(frame, list) else type(frame).__name__
            raise ValueError(
                f"SFT row{location} audio_codes frame {frame_index} must contain 16 tokens, got {width}"
            )
