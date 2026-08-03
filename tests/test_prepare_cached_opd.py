from __future__ import annotations

import unittest

from qwen3opsd.prepare_cached_opd import build_cached_opd_rows


class PrepareCachedOpdTest(unittest.TestCase):
    def test_pairs_different_rows_within_key_speaker(self) -> None:
        rows = [
            {
                "key": "xiaoyi/001",
                "text": "target one",
                "codes_path": "/codes/001.npy",
                "spk_emb_path": "/emb/001.npy",
                "caption": "warm voice",
            },
            {
                "key": "xiaoyi/002",
                "text": "target two",
                "codes_path": "/codes/002.npy",
                "spk_emb_path": "/emb/002.npy",
                "caption": "calm voice",
            },
        ]
        output = build_cached_opd_rows(rows)
        self.assertEqual(output[0]["sample_id"], "xiaoyi/001")
        self.assertEqual(output[0]["teacher_ref_codes_path"], "/codes/002.npy")
        self.assertEqual(output[0]["teacher_ref_spk_emb_path"], "/emb/002.npy")
        self.assertEqual(output[0]["teacher_ref_text"], "target two")
        self.assertEqual(output[1]["teacher_ref_codes_path"], "/codes/001.npy")

    def test_rejects_single_row_speaker(self) -> None:
        with self.assertRaisesRegex(ValueError, "only one row"):
            build_cached_opd_rows(
                [{"key": "solo/001", "text": "one", "codes_path": "one.npy", "spk_emb_path": "one.npy"}]
            )


if __name__ == "__main__":
    unittest.main()
