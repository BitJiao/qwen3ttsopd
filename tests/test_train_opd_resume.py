import unittest

from qwen3tts_opd.train_opd import _resolve_resume_step


class ResumeStepTest(unittest.TestCase):
    def test_infers_step_from_checkpoint_directory(self):
        self.assertEqual(_resolve_resume_step("checkpoints/run/step_3000", None), 3000)

    def test_requires_checkpoint_for_explicit_step(self):
        with self.assertRaisesRegex(ValueError, "requires --resume-from-checkpoint"):
            _resolve_resume_step(None, 3000)

    def test_rejects_step_that_disagrees_with_directory(self):
        with self.assertRaisesRegex(ValueError, "does not match"):
            _resolve_resume_step("checkpoints/run/step_3000", 2500)


if __name__ == "__main__":
    unittest.main()
