from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import Dataset

from qwen3opsd.instruction_utils import get_instruction, get_target_text


class InstructionSFTDataset(Dataset):
    """Raw caption/text rows plus precomputed target codec codes."""

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
        audio_codes = torch.tensor(item["audio_codes"], dtype=torch.long)
        if audio_codes.ndim != 2 or audio_codes.shape[1] != 16:
            raise ValueError(f"audio_codes must have shape [T, 16], got {tuple(audio_codes.shape)}")
        return {"sample": item, "audio_codes": audio_codes}

    @staticmethod
    def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "samples": [item["sample"] for item in batch],
            "audio_codes": [item["audio_codes"] for item in batch],
        }


class VoiceDesignSFTDataset(Dataset):
    """Caption/text rows and target codec codes; no speaker reference is used."""

    def __init__(self, data: list[dict[str, Any]]) -> None:
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.data[index]
        if not get_instruction(item):
            raise ValueError("VoiceDesign SFT row requires a non-empty caption/instruction")
        get_target_text(item)
        audio_codes = torch.tensor(item["audio_codes"], dtype=torch.long)
        if audio_codes.ndim != 2 or audio_codes.shape[1] != 16:
            raise ValueError(
                f"audio_codes must have shape [T, 16], got {tuple(audio_codes.shape)}"
            )
        return {"sample": item, "audio_codes": audio_codes}

    @staticmethod
    def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "samples": [item["sample"] for item in batch],
            "audio_codes": [item["audio_codes"] for item in batch],
        }
