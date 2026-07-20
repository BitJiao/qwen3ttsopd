from __future__ import annotations

import unittest

from qwen3opsd.compare_inference import _safe_sample_name, generate_candidate


class FakeTTS:
    def __init__(self) -> None:
        self.calls = []

    def generate_voice_clone(self, **kwargs):
        self.calls.append(("clone", kwargs))
        return [object()], 24000

    def generate_voice_design(self, **kwargs):
        self.calls.append(("design", kwargs))
        return [object()], 24000


class CompareInferenceTest(unittest.TestCase):
    def test_base_teacher_uses_full_icl(self) -> None:
        tts = FakeTTS()
        generate_candidate(
            tts,
            "base_icl",
            {
                "text": "target",
                "language": "Chinese",
                "teacher_ref_audio": "reference.wav",
                "teacher_ref_text": "reference transcript",
            },
            non_streaming_mode=True,
            gen_kwargs={"max_new_tokens": 32},
        )
        mode, kwargs = tts.calls[0]
        self.assertEqual(mode, "clone")
        self.assertEqual(kwargs["ref_audio"], "reference.wav")
        self.assertEqual(kwargs["ref_text"], "reference transcript")
        self.assertFalse(kwargs["x_vector_only_mode"])

    def test_both_vd_candidates_use_instruction(self) -> None:
        for candidate in ("student", "vd_teacher"):
            with self.subTest(candidate=candidate):
                tts = FakeTTS()
                generate_candidate(
                    tts,
                    candidate,
                    {"text": "target", "instruction": "calm voice"},
                    non_streaming_mode=True,
                    gen_kwargs={},
                )
                mode, kwargs = tts.calls[0]
                self.assertEqual(mode, "design")
                self.assertEqual(kwargs["instruct"], "calm voice")

    def test_sample_name_is_stable_and_path_safe(self) -> None:
        self.assertEqual(_safe_sample_name("scene/a:b", 7), "00007_scene_a_b")


if __name__ == "__main__":
    unittest.main()
