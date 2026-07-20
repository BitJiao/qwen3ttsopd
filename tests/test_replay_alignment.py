from __future__ import annotations

import unittest

from qwen3tts_opd.alignment import frame_prediction_slice, next_token_codec_mask


class SliceRecorder:
    def __init__(self) -> None:
        self.key = None

    def __getitem__(self, key):
        self.key = key
        return key


class ReplayAlignmentTest(unittest.TestCase):
    def test_codec_frames_use_preceding_hidden_states(self) -> None:
        self.assertEqual(frame_prediction_slice(prefill_length=8, num_frames=3), slice(7, 10))

    def test_invalid_lengths_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            frame_prediction_slice(prefill_length=0, num_frames=1)
        with self.assertRaises(ValueError):
            frame_prediction_slice(prefill_length=8, num_frames=-1)

    def test_sft_mask_drops_the_previous_sequence_position(self) -> None:
        mask = SliceRecorder()
        result = next_token_codec_mask(mask)
        self.assertEqual(result, (slice(None), slice(1, None)))


if __name__ == "__main__":
    unittest.main()
