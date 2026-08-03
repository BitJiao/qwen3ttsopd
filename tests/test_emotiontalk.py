from __future__ import annotations

import csv
import json
import tempfile
import unittest
import wave
from pathlib import Path

from qwen3opsd.emotiontalk import clean_transcript, convert, load_utterances, parse_emotiontalk_key


def write_wav(path: Path, sample_value: int, frames: int = 240) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(24000)
        handle.writeframes(int(sample_value).to_bytes(2, "little", signed=True) * frames)


class EmotionTalkConversionTest(unittest.TestCase):
    def test_key_and_transcript_parsing(self) -> None:
        speaker, scene, sequence, sample = parse_emotiontalk_key(
            "G00009/G00009_42/G00009_42_14/G00009_42_14_024.wav"
        )
        self.assertEqual((speaker, scene, sequence), ("14", "G00009/G00009_42", 24))
        self.assertEqual(sample, "G00009__G00009_42__G00009_42_14__G00009_42_14_024")
        self.assertEqual(clean_transcript("[over/]hello[/over] [interrupted]world"), "hello world")

    def test_cycle_has_no_target_reference_leakage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            audio_root = root / "audio"
            transcription = root / "transcription.csv"
            captions = root / "audio.csv"
            output = root / "output"
            keys = [
                "G00002/G00002_01/G00002_01_14/G00002_01_14_001",
                "G00002/G00002_02/G00002_02_14/G00002_02_14_001",
                "G00002/G00002_02/G00002_02_14/G00002_02_14_002",
                "G00002/G00002_02/G00002_02_14/G00002_02_14_010",
            ]
            for index, key in enumerate(keys):
                write_wav(audio_root / f"{key}.wav", index)
            with transcription.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["name", "emotion", "chinese"])
                writer.writeheader()
                for index, key in enumerate(keys):
                    writer.writerow({"name": key, "emotion": "neutral", "chinese": f"text {index}"})
            with captions.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["file_name", "emotion", "content"])
                writer.writeheader()
                for index, key in enumerate(keys):
                    writer.writerow(
                        {
                            "file_name": f"{key}.wav",
                            "emotion": "neutral",
                            "content": repr({"caption_1": f"instruction {index}"}),
                        }
                    )
            rows = load_utterances(transcription, captions, audio_root, "caption_1", True)
            summary = convert(rows, output, "skip", True, None, student_mode="voice_design")
            self.assertEqual(summary["sft_train"], 4)
            self.assertEqual(summary["opd_train"], 3)
            with (output / "opd_train.jsonl").open(encoding="utf-8") as handle:
                opd = [json.loads(line) for line in handle]
            self.assertTrue(all("student_spk_audio" not in row for row in opd))
            targets = {row["target_audio"] for row in opd}
            references = {row["teacher_ref_audio"] for row in opd}
            self.assertEqual(targets, references)
            for row in opd:
                self.assertNotEqual(row["target_audio"], row["teacher_ref_audio"])

            base_output = root / "base_output"
            base_summary = convert(rows, base_output, "skip", True, None, student_mode="base")
            self.assertEqual(base_summary["sft_train"], 3)
            self.assertEqual(base_summary["opd_train"], 3)
            with (base_output / "opd_train.jsonl").open(encoding="utf-8") as handle:
                base_opd = [json.loads(line) for line in handle]
            self.assertTrue(all(row.get("student_spk_audio") for row in base_opd))
            self.assertTrue(
                all(row["target_audio"] != row["student_spk_audio"] for row in base_opd)
            )


if __name__ == "__main__":
    unittest.main()
