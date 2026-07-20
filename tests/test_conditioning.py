from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from qwen3tts_opd.conditioning import teacher_icl_inputs, teacher_prompt_items, voice_design_inputs


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


class FakeTorch:
    long = "long"
    float32 = "float32"

    @staticmethod
    def as_tensor(value, dtype=None):
        return np.asarray(value), dtype


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
            {"text": "target", "caption": "bright young female voice"},
        )
        self.assertEqual(input_id, "target")
        self.assertEqual(instruct_id, "instruction:bright young female voice")

    def test_teacher_loads_cached_full_icl_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            codes_path = root / "codes.npy"
            embedding_path = root / "embedding.npy"
            np.save(codes_path, np.arange(48).reshape(3, 16))
            np.save(embedding_path, np.arange(8, dtype=np.float32).reshape(1, 8))
            with patch.dict(sys.modules, {"torch": FakeTorch}):
                item = teacher_prompt_items(
                    None,
                    {
                        "teacher_ref_text": "reference transcript",
                        "teacher_ref_codes_path": str(codes_path),
                        "teacher_ref_spk_emb_path": str(embedding_path),
                    },
                )[0]
            self.assertFalse(item.x_vector_only_mode)
            self.assertTrue(item.icl_mode)
            self.assertEqual(item.ref_code[0].shape, (3, 16))
            self.assertEqual(item.ref_spk_embedding[0].shape, (8,))


if __name__ == "__main__":
    unittest.main()
