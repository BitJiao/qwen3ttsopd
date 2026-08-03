from __future__ import annotations

from types import SimpleNamespace
import unittest

import torch

from qwen3tts_opd.alignment import frame_prediction_slice
from qwen3tts_opd.core import _replay_token_logits


class FakeEmbedding:
    def __call__(self, token_ids: torch.Tensor) -> torch.Tensor:
        return torch.zeros((*token_ids.shape, 4), dtype=torch.float32)


class FakeTalker:
    def __init__(self) -> None:
        self.config = SimpleNamespace(num_code_groups=1)
        self.code_predictor = SimpleNamespace(get_input_embeddings=lambda: [])
        self.selected_hidden = None

    def get_input_embeddings(self) -> FakeEmbedding:
        return FakeEmbedding()

    def __call__(self, *, inputs_embeds, attention_mask, output_hidden_states):
        length = inputs_embeds.shape[1]
        positions = torch.arange(length, dtype=torch.float32)
        logits = positions.view(1, length, 1).expand(1, length, 32).clone()
        hidden = positions.view(1, length, 1).expand(1, length, 4).clone()
        return SimpleNamespace(logits=logits, hidden_states=((hidden,),))

    def forward_sub_talker_finetune(self, codes, hidden_states):
        self.selected_hidden = hidden_states.detach().clone()
        return torch.zeros((codes.shape[0], 0, 1)), None


class ReplayAlignmentTest(unittest.TestCase):
    def test_codec_labels_and_hidden_states_use_the_same_one_step_shift(self) -> None:
        talker = FakeTalker()
        model = SimpleNamespace(
            talker=talker,
            config=SimpleNamespace(talker_config=SimpleNamespace(codec_eos_token_id=9)),
        )
        codes = torch.tensor([[4], [5]], dtype=torch.long)
        prefill = torch.zeros((1, 3, 4), dtype=torch.float32)
        pad = torch.zeros((1, 1, 4), dtype=torch.float32)

        replay = _replay_token_logits(model, codes, prefill, pad, pad)

        self.assertEqual(replay.codec_0_labels.tolist(), [4, 5, 9])
        self.assertEqual(replay.codec_0_logits[:, 0].tolist(), [2.0, 3.0, 4.0])
        self.assertEqual(replay.first_codebook[:, 0].tolist(), [2.0, 3.0])
        self.assertEqual(replay.eos_first_codebook[:, 0].tolist(), [4.0])
        self.assertEqual(talker.selected_hidden[:, 0].tolist(), [2.0, 3.0])

    def test_frame_prediction_slice(self) -> None:
        self.assertEqual(frame_prediction_slice(prefill_length=8, num_frames=3), slice(7, 10))

    def test_invalid_frame_lengths_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            frame_prediction_slice(prefill_length=0, num_frames=1)
        with self.assertRaises(ValueError):
            frame_prediction_slice(prefill_length=8, num_frames=-1)


if __name__ == "__main__":
    unittest.main()
