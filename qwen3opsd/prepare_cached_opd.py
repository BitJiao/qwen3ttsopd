from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from qwen3opsd.instruction_utils import get_target_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pair cached Qwen3-TTS codes/speaker embeddings into Base-ICL OPD JSONL."
    )
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument(
        "--speaker-field",
        default=None,
        help="Optional explicit speaker field. By default use speaker_id/speaker/spk or the first key path segment.",
    )
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}: {exc.msg}") from exc
            if not isinstance(row, dict):
                raise TypeError(f"row {line_number} must be a JSON object")
            rows.append(row)
    return rows


def _speaker_id(row: dict[str, Any], speaker_field: str | None, row_number: int) -> str:
    fields = [speaker_field] if speaker_field else ["speaker_id", "speaker", "spk"]
    for field in fields:
        if field and row.get(field) is not None and str(row[field]).strip():
            return str(row[field]).strip()
    key = row.get("key")
    if key is not None and str(key).strip():
        return str(key).replace("\\", "/").split("/", 1)[0].strip()
    requested = speaker_field or "speaker_id/speaker/spk/key"
    raise KeyError(f"row {row_number} cannot determine speaker from {requested}")


def _validate_source_row(row: dict[str, Any], row_number: int) -> None:
    get_target_text(row)
    for field in ("codes_path", "spk_emb_path"):
        if not row.get(field):
            raise KeyError(f"row {row_number} requires {field}")


def build_cached_opd_rows(
    rows: list[dict[str, Any]], *, speaker_field: str | None = None
) -> list[dict[str, Any]]:
    groups: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for row_number, row in enumerate(rows, start=1):
        _validate_source_row(row, row_number)
        groups[_speaker_id(row, speaker_field, row_number)].append((row_number, row))

    output: list[dict[str, Any]] = []
    for speaker, group in groups.items():
        if len(group) < 2:
            raise ValueError(f"speaker {speaker!r} has only one row; Base ICL requires a different reference row")
        for index, (row_number, target) in enumerate(group):
            target_text = get_target_text(target).strip()
            reference = None
            for offset in range(1, len(group)):
                candidate = group[(index + offset) % len(group)][1]
                if candidate["codes_path"] == target["codes_path"]:
                    continue
                if get_target_text(candidate).strip() == target_text:
                    continue
                reference = candidate
                break
            if reference is None:
                raise ValueError(
                    f"row {row_number} has no distinct-text reference for speaker {speaker!r}; "
                    "provide at least two different utterances"
                )

            paired = dict(target)
            paired.setdefault("sample_id", target.get("key", row_number - 1))
            paired["teacher_ref_key"] = reference.get("key")
            paired["teacher_ref_codes_path"] = reference["codes_path"]
            paired["teacher_ref_spk_emb_path"] = reference["spk_emb_path"]
            paired["teacher_ref_text"] = get_target_text(reference)
            output.append(paired)
    return output


def main() -> None:
    args = parse_args()
    rows = load_rows(args.input_jsonl)
    if not rows:
        raise ValueError(f"no rows loaded from {args.input_jsonl}")
    output = build_cached_opd_rows(rows, speaker_field=args.speaker_field)
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as handle:
        for row in output:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"input_rows": len(rows), "output_rows": len(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
