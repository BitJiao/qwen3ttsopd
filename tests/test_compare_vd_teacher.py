from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from qwen3opsd.compare_vd_teacher import DEFAULT_CASES, build_demo_cases, sample_seed
from qwen3opsd.gap_eval import HARD_CASES, build_blind_rows, build_gap_cases


class VoiceDesignTeacherComparisonTest(unittest.TestCase):
    def test_cases_use_caption_matched_reference_without_target_text_leak(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            audio = Path(temporary) / "reference.wav"
            audio.touch()
            rows = [
                {
                    "sample_id": sample_id,
                    "text": f"reference text {index}",
                    "instruction": f"caption {index}",
                    "target_audio": str(audio),
                    "language": "Chinese",
                }
                for index, (_, sample_id, _) in enumerate(DEFAULT_CASES)
            ]
            cases = build_demo_cases(rows)
            self.assertEqual(len(cases), len(DEFAULT_CASES))
            for case in cases:
                self.assertNotEqual(case["text"], case["teacher_ref_text"])
                self.assertTrue(case["instruction"].startswith("caption"))
                self.assertEqual(case["teacher_ref_audio"], str(audio.resolve()))

    def test_sample_seed_is_stable_and_sample_specific(self) -> None:
        self.assertEqual(sample_seed(1, "a"), sample_seed(1, "a"))
        self.assertNotEqual(sample_seed(1, "a"), sample_seed(1, "b"))

    def test_hard_gap_cases_and_blinding(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference = root / "reference.wav"
            reference.touch()
            rows = [
                {
                    "sample_id": sample_id,
                    "text": f"reference {index}",
                    "instruction": f"caption {index}",
                    "target_audio": str(reference),
                    "emotion": emotion,
                }
                for index, (_, sample_id, _) in enumerate(HARD_CASES)
                for emotion in ["expressive"]
            ]
            cases = build_gap_cases(rows)
            for index, case in enumerate(cases):
                student = root / f"student_{index}.wav"
                teacher = root / f"teacher_{index}.wav"
                student.touch()
                teacher.touch()
                case["student_vd_audio"] = str(student)
                case["teacher_icl_audio"] = str(teacher)
                self.assertNotEqual(case["text"], case["teacher_ref_text"])
                self.assertNotIn("caption", case["teacher_conditioning"])
            blind_rows, key = build_blind_rows(cases, root, seed=7)
            self.assertEqual(len(blind_rows), len(HARD_CASES))
            self.assertEqual(
                {model for mapping in key.values() for model in mapping.values()},
                {"VD Student", "ICL Teacher"},
            )


if __name__ == "__main__":
    unittest.main()
