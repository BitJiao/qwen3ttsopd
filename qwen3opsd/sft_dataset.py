from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import Dataset

from qwen3opsd.data_contract import load_audio_codes
from qwen3opsd.instruction_utils import get_instruction, get_target_text


def _target_codes(item: dict[str, Any], index: int) -> torch.Tensor:
    return torch.as_tensor(
        load_audio_codes(item, row_number=index + 1),
        dtype=torch.long,
    )


class InstructionSFTDataset(Dataset):
    """Base-model caption/text rows plus target codec codes and enrollment."""

    def __init__(self, data: list[dict[str, Any]], processor=None, config=None) -> None:
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.data[index]
        if not get_instruction(item):
            raise ValueError("SFT row requires a non-empty caption/instruction")
        get_target_text(item)
        student_spk_audio = item.get("student_spk_audio", item.get("ref_audio"))
        if not student_spk_audio:
            raise KeyError("SFT row requires student_spk_audio (or legacy ref_audio)")
        return {"sample": item, "audio_codes": _target_codes(item, index)}

    @staticmethod
    def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "samples": [item["sample"] for item in batch],
            "audio_codes": [item["audio_codes"] for item in batch],
        }


class VoiceDesignSFTDataset(Dataset):
    """VoiceDesign caption/text rows plus target codec codes; no enrollment."""

    def __init__(self, data: list[dict[str, Any]]) -> None:
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.data[index]
        if not get_instruction(item):
            raise ValueError("VoiceDesign SFT row requires a non-empty caption/instruction")
        get_target_text(item)
        return {"sample": item, "audio_codes": _target_codes(item, index)}

    @staticmethod
    def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "samples": [item["sample"] for item in batch],
            "audio_codes": [item["audio_codes"] for item in batch],
        }
