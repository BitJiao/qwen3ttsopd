from __future__ import annotations

import unittest

from qwen3tts_opd.alignment import frame_prediction_slice


class ReplayAlignmentTest(unittest.TestCase):
    def test_codec_frames_use_preceding_hidden_states(self) -> None:
        self.assertEqual(frame_prediction_slice(prefill_length=8, num_frames=3), slice(7, 10))

    def test_invalid_lengths_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            frame_prediction_slice(prefill_length=0, num_frames=1)
        with self.assertRaises(ValueError):
            frame_prediction_slice(prefill_length=8, num_frames=-1)


if __name__ == "__main__":
    unittest.main()
