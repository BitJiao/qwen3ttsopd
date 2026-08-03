from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from qwen3opsd.four_way_eval import build_named_rows, validate_cases


class FourWayEvaluationTest(unittest.TestCase):
    def test_named_rows_keep_fixed_system_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = {}
            for field in (
                "teacher_ref_audio",
                "student_vd_audio",
                "sft_vd_audio",
                "teacher_icl_audio",
                "teacher_icl_caption_audio",
            ):
                path = root / f"{field}.wav"
                path.touch()
                paths[field] = str(path)
            cases = [
                {
                    "sample_id": "case",
                    "text": "target",
                    "instruction": "caption",
                    "teacher_ref_text": "words spoken in reference",
                    **paths,
                }
            ]
            validate_cases(cases)
            rows = build_named_rows(cases, root)
            self.assertEqual(
                [system["model"] for system in rows[0]["systems"]],
                [
                    "Original VoiceDesign",
                    "SFT VoiceDesign",
                    "Base ICL",
                    "Base ICL + Caption",
                ],
            )
            self.assertIn(
                "reference transcript", rows[0]["systems"][3]["conditioning"]
            )


if __name__ == "__main__":
    unittest.main()
