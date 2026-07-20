from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import Dataset

from qwen3opsd.data_contract import load_audio_codes


class VoiceDesignSFTDataset(Dataset):
    """VoiceDesign instruction SFT rows with precomputed target codec codes."""

    def __init__(self, data: list[dict[str, Any]]) -> None:
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = dict(self.data[index])
        item["audio_codes"] = torch.as_tensor(load_audio_codes(item, row_number=index + 1), dtype=torch.long)
        return item

    @staticmethod
    def collate_fn(batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return batch
