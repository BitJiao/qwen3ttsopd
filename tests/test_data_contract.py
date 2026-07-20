from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from qwen3opsd.prepare_codes import get_target_audio, load_rows


class DataContractTest(unittest.TestCase):
    def test_target_audio_does_not_require_legacy_audio_field(self) -> None:
        self.assertEqual(get_target_audio({"target_audio": "target.wav"}), "target.wav")
        self.assertEqual(get_target_audio({"audio": "legacy.wav"}), "legacy.wav")

    def test_missing_target_audio_has_row_number(self) -> None:
        with self.assertRaisesRegex(KeyError, "row 7"):
            get_target_audio({}, row_number=7)

    def test_jsonl_requires_one_object_per_line(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "train.jsonl"
            path.write_text(json.dumps([{"target_audio": "target.wav"}]) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(TypeError, "each JSONL line must be an object"):
                load_rows(path)


if __name__ == "__main__":
    unittest.main()
