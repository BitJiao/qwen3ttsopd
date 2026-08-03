from __future__ import annotations

import unittest

from qwen3tts_opd.teacher_modes import get_teacher_mode, validate_opd_row


class TeacherModeTest(unittest.TestCase):
    def test_base_icl_requires_reference_audio_and_text(self) -> None:
        with self.assertRaisesRegex(KeyError, "teacher_ref_text"):
            validate_opd_row({"text": "target"}, "base_icl", row_number=1)

    def test_voice_design_teacher_needs_no_icl_fields(self) -> None:
        validate_opd_row(
            {"text": "target", "instruction": "warm and calm voice"},
            "voice_design",
            row_number=1,
        )
        self.assertEqual(get_teacher_mode("voice_design").conditioning, "voice_design")

    def test_base_icl_accepts_cached_reference(self) -> None:
        validate_opd_row(
            {
                "text": "target",
                "codes_path": "target.npy",
                "spk_emb_path": "target_emb.npy",
                "teacher_ref_text": "reference",
                "teacher_ref_codes_path": "reference.npy",
                "teacher_ref_spk_emb_path": "reference_emb.npy",
            },
            "base_icl",
            row_number=1,
        )

    def test_cached_reference_cannot_be_target(self) -> None:
        with self.assertRaisesRegex(ValueError, "identical codes_path"):
            validate_opd_row(
                {
                    "text": "target",
                    "codes_path": "same.npy",
                    "spk_emb_path": "target_emb.npy",
                    "teacher_ref_text": "reference",
                    "teacher_ref_codes_path": "same.npy",
                    "teacher_ref_spk_emb_path": "reference_emb.npy",
                },
                "base_icl",
            )


if __name__ == "__main__":
    unittest.main()
