from __future__ import annotations

import unittest

from qwen3tts_opd.teacher_modes import get_teacher_mode, validate_opd_row


class TeacherModeTest(unittest.TestCase):
    def test_base_icl_requires_reference_audio_and_text(self) -> None:
        with self.assertRaisesRegex(KeyError, "teacher_ref_audio"):
            validate_opd_row({"text": "target"}, "base_icl", row_number=1)

    def test_voice_design_teacher_needs_no_icl_fields(self) -> None:
        validate_opd_row(
            {"text": "target", "instruction": "warm and calm voice"},
            "voice_design",
            row_number=1,
        )
        self.assertEqual(get_teacher_mode("voice_design").conditioning, "voice_design")


if __name__ == "__main__":
    unittest.main()
