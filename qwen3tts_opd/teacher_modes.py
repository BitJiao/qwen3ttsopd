from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from qwen3tts_opd.instruction_utils import get_target_text


@dataclass(frozen=True)
class TeacherMode:
    name: str
    model_type: str
    conditioning: str
    requires_icl: bool


TEACHER_MODES = {
    "base_icl": TeacherMode(
        name="base_icl",
        model_type="base",
        conditioning="teacher_icl",
        requires_icl=True,
    ),
    "voice_design": TeacherMode(
        name="voice_design",
        model_type="voice_design",
        conditioning="voice_design",
        requires_icl=False,
    ),
}


def get_teacher_mode(name: str) -> TeacherMode:
    try:
        return TEACHER_MODES[name]
    except KeyError as exc:
        raise ValueError(f"unknown teacher mode {name!r}; choose from {sorted(TEACHER_MODES)}") from exc


def validate_opd_row(row: dict[str, Any], teacher_mode: str, *, row_number: int | None = None) -> None:
    location = f" {row_number}" if row_number is not None else ""
    try:
        get_target_text(row)
    except KeyError as exc:
        raise KeyError(f"OPD row{location} requires text (or target_text)") from exc

    mode = get_teacher_mode(teacher_mode)
    if not mode.requires_icl:
        return

    teacher_audio = row.get("teacher_ref_audio", row.get("ref_audio"))
    teacher_codes = row.get("teacher_ref_codes_path", row.get("ref_codes_path"))
    teacher_spk_emb = row.get("teacher_ref_spk_emb_path", row.get("ref_spk_emb_path"))
    teacher_text = row.get("teacher_ref_text", row.get("ref_text"))
    if not teacher_text:
        raise KeyError(f"OPD row{location} requires teacher_ref_text (or legacy ref_text) for teacher ICL")
    if teacher_codes or teacher_spk_emb:
        if not teacher_codes:
            raise KeyError(f"OPD row{location} cached ICL requires teacher_ref_codes_path")
        if not teacher_spk_emb:
            raise KeyError(f"OPD row{location} cached ICL requires teacher_ref_spk_emb_path")
    elif not teacher_audio:
        raise KeyError(
            f"OPD row{location} requires teacher_ref_audio, or cached "
            "teacher_ref_codes_path and teacher_ref_spk_emb_path"
        )
    target_audio = row.get("target_audio", row.get("audio"))
    if target_audio and teacher_audio and os.path.realpath(target_audio) == os.path.realpath(teacher_audio):
        raise ValueError(f"OPD row{location} has identical target_audio and teacher_ref_audio")
    target_codes = row.get("codes_path")
    if target_codes and teacher_codes and os.path.realpath(target_codes) == os.path.realpath(teacher_codes):
        raise ValueError(f"OPD row{location} has identical codes_path and teacher_ref_codes_path")
