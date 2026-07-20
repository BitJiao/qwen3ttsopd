from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import Dataset


def validate_sft_row(item: dict[str, Any], *, row_number: int | None = None) -> None:
    location = f" {row_number}" if row_number is not None else ""
    student_spk_audio = item.get("student_spk_audio") or item.get("ref_audio")
    if not student_spk_audio:
        raise KeyError(f"SFT row{location} requires student_spk_audio (or legacy ref_audio)")
    if "audio_codes" not in item:
        raise KeyError(f"SFT row{location} requires audio_codes; run prepare_sft.sh first")
    audio_codes = torch.as_tensor(item["audio_codes"])
    if audio_codes.ndim != 2 or audio_codes.shape[1] != 16:
        raise ValueError(
            f"SFT row{location} audio_codes must have shape [T, 16], got {tuple(audio_codes.shape)}"
        )
    if audio_codes.shape[0] == 0:
        raise ValueError(f"SFT row{location} audio_codes must contain at least one frame")


class InstructionSFTDataset(Dataset):
    """Qwen3-TTS teacher-forcing dataset with per-speaker enrollment audio."""

    def __init__(self, data: list[dict[str, Any]], processor, config) -> None:
        self.data = data
        self.processor = processor
        self.config = config

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.data[index]
        text = f"<|im_start|>assistant\n{item['text']}<|im_end|>\n<|im_start|>assistant\n"
        text_ids = self.processor(text=text, return_tensors="pt", padding=True)["input_ids"]
        if text_ids.ndim == 1:
            text_ids = text_ids.unsqueeze(0)
        student_spk_audio = item.get("student_spk_audio") or item.get("ref_audio")
        return {
            "text_ids": text_ids[:, :-5],
            "audio_codes": torch.tensor(item["audio_codes"], dtype=torch.long),
            "student_spk_audio": str(student_spk_audio),
        }

    def collate_fn(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        lengths = [item["text_ids"].shape[1] + item["audio_codes"].shape[0] for item in batch]
        batch_size = len(batch)
        max_length = max(lengths) + 8
        input_ids = torch.zeros((batch_size, max_length, 2), dtype=torch.long)
        codec_ids = torch.zeros((batch_size, max_length, 16), dtype=torch.long)
        text_embedding_mask = torch.zeros((batch_size, max_length), dtype=torch.bool)
        codec_embedding_mask = torch.zeros((batch_size, max_length), dtype=torch.bool)
        codec_mask = torch.zeros((batch_size, max_length), dtype=torch.bool)
        attention_mask = torch.zeros((batch_size, max_length), dtype=torch.long)
        codec_0_labels = torch.full((batch_size, max_length), -100, dtype=torch.long)

        for row_index, item in enumerate(batch):
            text_ids = item["text_ids"]
            audio_codes = item["audio_codes"]
            if audio_codes.ndim != 2 or audio_codes.shape[1] != 16:
                raise ValueError(f"audio_codes must have shape [T, 16], got {tuple(audio_codes.shape)}")
            codec_0 = audio_codes[:, 0]
            text_len = text_ids.shape[1]
            codec_len = codec_0.shape[0]

            input_ids[row_index, :3, 0] = text_ids[0, :3]
            input_ids[row_index, 3:7, 0] = self.config.tts_pad_token_id
            input_ids[row_index, 7, 0] = self.config.tts_bos_token_id
            input_ids[row_index, 8 : 8 + text_len - 3, 0] = text_ids[0, 3:]
            input_ids[row_index, 8 + text_len - 3, 0] = self.config.tts_eos_token_id
            input_ids[row_index, 8 + text_len - 2 : 8 + text_len + codec_len, 0] = self.config.tts_pad_token_id
            text_embedding_mask[row_index, : 8 + text_len + codec_len] = True

            input_ids[row_index, 3:8, 1] = torch.tensor(
                [
                    self.config.talker_config.codec_nothink_id,
                    self.config.talker_config.codec_think_bos_id,
                    self.config.talker_config.codec_think_eos_id,
                    0,
                    self.config.talker_config.codec_pad_id,
                ]
            )
            input_ids[row_index, 8 : 8 + text_len - 2, 1] = self.config.talker_config.codec_pad_id
            input_ids[row_index, 8 + text_len - 2, 1] = self.config.talker_config.codec_bos_id
            codec_start = 8 + text_len - 1
            input_ids[row_index, codec_start : codec_start + codec_len, 1] = codec_0
            input_ids[row_index, codec_start + codec_len, 1] = self.config.talker_config.codec_eos_token_id

            codec_0_labels[row_index, codec_start : codec_start + codec_len] = codec_0
            codec_0_labels[row_index, codec_start + codec_len] = self.config.talker_config.codec_eos_token_id
            codec_ids[row_index, codec_start : codec_start + codec_len] = audio_codes
            codec_embedding_mask[row_index, 3 : 8 + text_len + codec_len] = True
            codec_embedding_mask[row_index, 6] = False
            codec_mask[row_index, codec_start : codec_start + codec_len] = True
            attention_mask[row_index, : 8 + text_len + codec_len] = True

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "text_embedding_mask": text_embedding_mask.unsqueeze(-1),
            "codec_embedding_mask": codec_embedding_mask.unsqueeze(-1),
            "codec_0_labels": codec_0_labels,
            "codec_ids": codec_ids,
            "codec_mask": codec_mask,
            "student_spk_audio": [item["student_spk_audio"] for item in batch],
        }
