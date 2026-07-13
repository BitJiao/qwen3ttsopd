from __future__ import annotations

import unittest

from qwen3tts_opd.core import _prompt_for_sample


class FakePrompt:
    def __init__(self, ref_text):
        self.ref_text = ref_text


class FakeTTS:
    def __init__(self) -> None:
        self.calls = []

    def create_voice_clone_prompt(self, **kwargs):
        self.calls.append(kwargs)
        return [FakePrompt(kwargs["ref_text"][0])]

    def _prompt_items_to_voice_clone_prompt(self, items):
        return {"items": items}

    def _build_assistant_text(self, text):
        return text

    def _build_ref_text(self, text):
        return text

    def _tokenize_texts(self, texts):
        return [texts[0]]


class ConditioningContractTest(unittest.TestCase):
    def test_student_and_teacher_use_different_references(self) -> None:
        sample = {
            "text": "target",
            "student_spk_audio": "enrollment.wav",
            "teacher_ref_audio": "teacher.wav",
            "teacher_ref_text": "teacher transcript",
        }
        tts = FakeTTS()
        _prompt_for_sample(tts, sample, x_vector_only_mode=True)
        _prompt_for_sample(tts, sample, x_vector_only_mode=False)
        self.assertEqual(tts.calls[0]["ref_audio"], ["enrollment.wav"])
        self.assertEqual(tts.calls[0]["ref_text"], [None])
        self.assertEqual(tts.calls[1]["ref_audio"], ["teacher.wav"])
        self.assertEqual(tts.calls[1]["ref_text"], ["teacher transcript"])


if __name__ == "__main__":
    unittest.main()
