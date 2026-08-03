from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from qwen3opsd.build_eval_report import build_report_rows, parse_run_spec, write_gemini_jsonls
from qwen3opsd.eval_emotiontalk import (
    conditioned_text,
    latest_manifest_rows,
    safe_filename,
    select_rows,
    stable_sample_seed,
    validate_rows,
)
from qwen3opsd.prepare_instructttseval import infer_gender, select_task_indices


class EmotionTalkEvaluationTest(unittest.TestCase):
    def test_conditioning_and_stable_seed(self) -> None:
        row = {"text": "你好。", "instruction": "语速稍快。"}
        self.assertEqual(conditioned_text(row, "text_only"), "你好。")
        self.assertEqual(conditioned_text(row, "instruction"), "你好。")
        self.assertEqual(stable_sample_seed(7, "sample"), stable_sample_seed(7, "sample"))
        self.assertNotEqual(stable_sample_seed(7, "sample"), stable_sample_seed(7, "other"))

    def test_validation_selection_and_filename(self) -> None:
        rows = [
            {
                "sample_id": "a/b",
                "text": "x",
                "instruction": "y",
                "student_spk_audio": "speaker.wav",
                "target_audio": "target.wav",
            },
            {
                "sample_id": "c",
                "text": "x",
                "instruction": "y",
                "student_spk_audio": "speaker.wav",
                "target_audio": "target.wav",
            },
        ]
        validate_rows(rows)
        self.assertEqual(select_rows(rows, 1, 1), rows[1:])
        self.assertEqual(safe_filename("a/b"), "a_b")

    def test_manifest_uses_latest_valid_line(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "manifest.jsonl"
            path.write_text(
                json.dumps({"sample_id": "a", "status": "error"})
                + "\n"
                + "{incomplete\n"
                + json.dumps({"sample_id": "a", "status": "ok"})
                + "\n",
                encoding="utf-8",
            )
            self.assertEqual(latest_manifest_rows(path)["a"]["status"], "ok")

    def test_report_intersects_runs_and_blinds_labels(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = []
            for sample_id in ("one", "two"):
                target = root / f"{sample_id}_target.wav"
                enrollment = root / f"{sample_id}_enrollment.wav"
                target.touch()
                enrollment.touch()
                source.append(
                    {
                        "sample_id": sample_id,
                        "text": "text",
                        "instruction": "instruction",
                        "target_audio": str(target),
                        "student_spk_audio": str(enrollment),
                    }
                )

            runs = []
            for label in ("Base", "SFT", "OPD"):
                manifest = root / f"{label}.jsonl"
                lines = []
                for sample_id in ("one", "two"):
                    audio = root / f"{label}_{sample_id}.wav"
                    audio.touch()
                    lines.append(
                        json.dumps(
                            {
                                "sample_id": sample_id,
                                "status": "ok",
                                "model_name": label,
                                "conditioning": "instruction",
                                "generated_audio": str(audio),
                            }
                        )
                    )
                manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
                runs.append((label, manifest))

            rows, key, counts = build_report_rows(source, runs, root, seed=4, limit=1)
            self.assertEqual(len(rows), 1)
            self.assertEqual(counts, {"Base": 2, "SFT": 2, "OPD": 2})
            self.assertEqual(set(key["one"]), {"A", "B", "C"})
            self.assertEqual({item["system_id"] for item in rows[0]["systems"]}, {"A", "B", "C"})

            rows[0]["source_id"] = "source"
            rows[0]["task"] = "APS"
            gemini = write_gemini_jsonls(root, rows)
            self.assertEqual(set(gemini), {"Base", "SFT", "OPD"})
            record = json.loads((root / "gemini_Base.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(record["APS"]["instruction"], "instruction")

    def test_parse_run_spec(self) -> None:
        label, path = parse_run_spec("Base=manifest.jsonl")
        self.assertEqual(label, "Base")
        self.assertEqual(path.name, "manifest.jsonl")

    def test_instructttseval_gender_and_selection(self) -> None:
        self.assertEqual(infer_gender("性别: 女性。", "zh"), "female")
        self.assertEqual(infer_gender("gender: Male.", "en"), "male")
        self.assertEqual(infer_gender("性别: 先男声后女声。", "zh"), "male")
        self.assertEqual(infer_gender("性别: 女童。", "zh"), "female")
        selection = select_task_indices(100, 10, 7)
        self.assertEqual({key: len(value) for key, value in selection.items()}, {"APS": 10, "DSD": 10, "RP": 10})
        self.assertEqual(len(set().union(*map(set, selection.values()))), 30)


if __name__ == "__main__":
    unittest.main()
