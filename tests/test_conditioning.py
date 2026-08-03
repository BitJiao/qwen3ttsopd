from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from qwen3opsd.sft_dataset import InstructionSFTDataset, VoiceDesignSFTDataset
from qwen3tts_opd.conditioning import teacher_icl_inputs, teacher_prompt_items, voice_design_inputs
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
        return f"assistant:{text}"

    def _build_instruct_text(self, text):
        return f"user:{text}"

    def _build_ref_text(self, text):
        return text

    def _tokenize_texts(self, texts):
        return [texts[0]]


class FakeTorch:
    long = "long"
    float32 = "float32"

    @staticmethod
    def as_tensor(value, dtype=None):
        return np.asarray(value), dtype


class ConditioningContractTest(unittest.TestCase):
    def test_student_and_teacher_use_different_references(self) -> None:
        sample = {
            "text": "target",
            "instruction": "caption",
            "student_spk_audio": "enrollment.wav",
            "teacher_ref_audio": "teacher.wav",
            "teacher_ref_text": "teacher transcript",
        }
        tts = FakeTTS()
        student = _prompt_for_sample(tts, sample, x_vector_only_mode=True)
        teacher = _prompt_for_sample(tts, sample, x_vector_only_mode=False)
        self.assertEqual(tts.calls[0]["ref_audio"], ["enrollment.wav"])
        self.assertEqual(tts.calls[0]["ref_text"], [None])
        self.assertEqual(tts.calls[1]["ref_audio"], ["teacher.wav"])
        self.assertEqual(tts.calls[1]["ref_text"], ["teacher transcript"])
        self.assertEqual(student[0], "assistant:target")
        self.assertEqual(student[1], "user:caption")
        self.assertEqual(teacher[0], "assistant:target")
        self.assertEqual(teacher[1], "user:caption")

    def test_teacher_uses_privileged_icl_reference(self) -> None:
        tts = FakeTTS()
        teacher_icl_inputs(
            tts,
            {
                "text": "target",
                "teacher_ref_audio": "teacher.wav",
                "teacher_ref_text": "teacher transcript",
            },
        )
        self.assertEqual(tts.calls[0]["ref_audio"], ["teacher.wav"])
        self.assertEqual(tts.calls[0]["ref_text"], ["teacher transcript"])
        self.assertEqual(tts.calls[0]["x_vector_only_mode"], [False])

    def test_voice_design_uses_separate_instruction(self) -> None:
        input_id, instruct_id = voice_design_inputs(
            FakeTTS(),
            {"text": "target", "caption": "bright young female voice"},
        )
        self.assertEqual(input_id, "assistant:target")
        self.assertEqual(instruct_id, "user:bright young female voice")

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

    def test_sft_tokenizes_caption_and_target_separately(self) -> None:
        dataset = InstructionSFTDataset(
            [{
                "text": "target",
                "instruction": "caption",
                "audio_codes": [[0] * 16],
                "student_spk_audio": "enrollment.wav",
            }]
        )
        item = dataset[0]
        self.assertEqual(item["sample"]["text"], "target")
        self.assertEqual(item["sample"]["instruction"], "caption")
        self.assertEqual(tuple(item["audio_codes"].shape), (1, 16))

    def test_voice_design_sft_does_not_require_reference_audio(self) -> None:
        dataset = VoiceDesignSFTDataset(
            [{"text": "target", "instruction": "caption", "audio_codes": [[0] * 16]}]
        )
        item = dataset[0]
        self.assertNotIn("student_spk_audio", item["sample"])
        self.assertEqual(tuple(item["audio_codes"].shape), (1, 16))


if __name__ == "__main__":
    unittest.main()
