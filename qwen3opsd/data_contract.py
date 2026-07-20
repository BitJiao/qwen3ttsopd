from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from qwen3opsd.instruction_utils import get_target_text


def _normalize_audio_codes(codes: Any, *, location: str = "") -> np.ndarray:
    value = np.asarray(codes)
    if value.ndim == 3 and value.shape[0] == 1:
        value = value[0]
    if value.ndim == 2 and value.shape[1] != 16 and value.shape[0] == 16:
        value = value.T
    if value.ndim != 2 or value.shape[0] == 0 or value.shape[1] != 16:
        raise ValueError(f"SFT row{location} audio codes must have shape [T, 16], got {value.shape}")
    return value


def load_audio_codes(item: dict[str, Any], *, row_number: int | None = None) -> np.ndarray:
    location = f" {row_number}" if row_number is not None else ""
    if "audio_codes" in item:
        codes = np.asarray(item["audio_codes"], dtype=np.int64)
    else:
        codes_path = item.get("codes_path")
        if not codes_path:
            raise KeyError(f"SFT row{location} requires audio_codes or codes_path")
        path = Path(str(codes_path))
        if not path.is_file():
            raise FileNotFoundError(f"SFT row{location} codes_path does not exist: {path}")
        codes = np.asarray(np.load(path, allow_pickle=False))
    return _normalize_audio_codes(codes, location=location).astype(np.int64, copy=False)


def validate_sft_row(item: dict[str, Any], *, row_number: int | None = None) -> None:
    location = f" {row_number}" if row_number is not None else ""
    try:
        get_target_text(item)
    except KeyError as exc:
        raise KeyError(f"SFT row{location} requires text (or target_text)") from exc
    if "audio_codes" in item:
        _normalize_audio_codes(item["audio_codes"], location=location)
    elif not item.get("codes_path"):
        raise KeyError(f"SFT row{location} requires audio_codes or codes_path")
