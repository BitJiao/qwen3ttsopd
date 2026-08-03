from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from qwen3opsd.portable_manifest import convert_directory


class PortableManifestTest(unittest.TestCase):
    def _write_jsonl(self, path: Path, rows: list[dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )

    def test_export_and_materialize_nested_audio_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_audio = root / "source_audio"
            target_audio = root / "target_audio"
            relative_paths = (Path("Audio/a.wav"), Path("Audio/b.wav"))
            for audio_root in (source_audio, target_audio):
                for relative in relative_paths:
                    path = audio_root / relative
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(b"wav")

            rows = [
                {
                    "audio": str(source_audio / relative_paths[0]),
                    "student_spk_audio": str(source_audio / relative_paths[1]),
                    "cycle": [
                        {"teacher_ref_audio": str(source_audio / relative_paths[1])}
                    ],
                }
            ]
            source = root / "source"
            portable = root / "portable"
            materialized = root / "materialized"
            self._write_jsonl(source / "rows.jsonl", rows)

            convert_directory(
                source,
                portable,
                source_audio,
                mode="export",
                files=["rows.jsonl"],
            )
            exported = json.loads((portable / "rows.jsonl").read_text().strip())
            self.assertEqual(exported["audio"], "Audio/a.wav")
            self.assertEqual(exported["student_spk_audio"], "Audio/b.wav")
            self.assertEqual(exported["cycle"][0]["teacher_ref_audio"], "Audio/b.wav")

            convert_directory(
                portable,
                materialized,
                target_audio,
                mode="materialize",
                files=["rows.jsonl"],
                check_audio=True,
            )
            restored = json.loads((materialized / "rows.jsonl").read_text().strip())
            self.assertEqual(restored["audio"], str((target_audio / "Audio/a.wav").resolve()))
            self.assertEqual(
                restored["cycle"][0]["teacher_ref_audio"],
                str((target_audio / "Audio/b.wav").resolve()),
            )

    def test_export_rejects_audio_outside_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            self._write_jsonl(source / "rows.jsonl", [{"audio": str(root / "outside.wav")}])
            with self.assertRaisesRegex(ValueError, "outside --audio-root"):
                convert_directory(
                    source,
                    root / "portable",
                    root / "audio",
                    mode="export",
                    files=["rows.jsonl"],
                )


if __name__ == "__main__":
    unittest.main()
