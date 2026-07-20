from __future__ import annotations

import unittest

from qwen3tts_opd.conditioning import teacher_icl_inputs, voice_design_inputs


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

    def _build_instruct_text(self, text):
        return f"instruction:{text}"

    def _tokenize_texts(self, texts):
        return [texts[0]]


class ConditioningContractTest(unittest.TestCase):
    def test_teacher_uses_privileged_icl_reference(self) -> None:
        sample = {
            "text": "target",
            "teacher_ref_audio": "teacher.wav",
            "teacher_ref_text": "teacher transcript",
        }
        tts = FakeTTS()
        teacher_icl_inputs(tts, sample)
        self.assertEqual(tts.calls[0]["ref_audio"], ["teacher.wav"])
        self.assertEqual(tts.calls[0]["ref_text"], ["teacher transcript"])
        self.assertEqual(tts.calls[0]["x_vector_only_mode"], [False])

    def test_student_uses_separate_voice_design_instruction(self) -> None:
        tts = FakeTTS()
        input_id, instruct_id = voice_design_inputs(
            tts,
            {"text": "target", "instruction": "bright young female voice"},
        )
        self.assertEqual(input_id, "target")
        self.assertEqual(instruct_id, "instruction:bright young female voice")


if __name__ == "__main__":
    unittest.main()
