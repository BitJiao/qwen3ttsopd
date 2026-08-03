from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from qwen3opsd.three_way_eval import build_blind_rows, validate_baseline_cases


class ThreeWayEvaluationTest(unittest.TestCase):
    def test_blinding_contains_all_three_models(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = []
            for name in ("ref.wav", "base.wav", "sft.wav", "icl.wav"):
                path = root / name
                path.touch()
                paths.append(str(path))
            cases = [
                {
                    "sample_id": "case",
                    "emotion": "angry",
                    "text": "target",
                    "instruction": "caption",
                    "teacher_ref_text": "reference",
                    "teacher_ref_audio": paths[0],
                    "student_vd_audio": paths[1],
                    "sft_vd_audio": paths[2],
                    "teacher_icl_audio": paths[3],
                }
            ]
            validate_baseline_cases(cases)
            rows, key = build_blind_rows(cases, root, 7)
            self.assertEqual(len(rows[0]["systems"]), 3)
            self.assertEqual(
                set(key["case"].values()),
                {"Original VoiceDesign", "SFT VoiceDesign", "Base ICL"},
            )


if __name__ == "__main__":
    unittest.main()
