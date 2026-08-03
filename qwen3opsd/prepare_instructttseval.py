from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any


TASKS = ("APS", "DSD", "RP")


def infer_gender(aps_instruction: str, language: str) -> str:
    text = str(aps_instruction)
    if language == "zh":
        female_match = re.search(r"女性|女声|女人|女孩|少女|女童", text)
        male_match = re.search(r"男性|男声|男人|男孩|男童", text)
    else:
        female_match = re.search(r"\bfemale\b|\bwoman\b|\bgirl\b", text, re.IGNORECASE)
        male_match = re.search(r"\bmale\b|\bman\b|\bboy\b", text, re.IGNORECASE)
    if female_match is None and male_match is None:
        raise ValueError(f"cannot infer one gender from APS instruction: {text[:160]!r}")
    if female_match is None:
        return "male"
    if male_match is None:
        return "female"
    return "female" if female_match.start() < male_match.start() else "male"


def select_task_indices(row_count: int, num_per_task: int, seed: int) -> dict[str, list[int]]:
    if row_count <= 0:
        raise ValueError("row_count must be positive")
    if num_per_task < 0:
        raise ValueError("num_per_task must be non-negative")
    if num_per_task == 0:
        return {task: list(range(row_count)) for task in TASKS}
    if num_per_task * len(TASKS) > row_count:
        raise ValueError("not enough source rows to select disjoint task subsets")
    indices = list(range(row_count))
    random.Random(seed).shuffle(indices)
    return {
        task: indices[offset * num_per_task : (offset + 1) * num_per_task]
        for offset, task in enumerate(TASKS)
    }


def extract_reference_audio(audio: dict[str, Any], output_path: Path) -> None:
    payload = audio.get("bytes") if audio else None
    if payload is None:
        source_path = audio.get("path") if audio else None
        if not source_path:
            raise ValueError("reference_audio contains neither bytes nor path")
        payload = Path(source_path).read_bytes()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not output_path.exists() or output_path.stat().st_size != len(payload):
        temporary = output_path.with_suffix(".tmp.wav")
        temporary.write_bytes(payload)
        temporary.replace(output_path)


def prepare_rows(
    source_rows: list[dict[str, Any]],
    selections: dict[str, list[int]],
    *,
    language: str,
    output_dir: Path,
    male_enrollment: Path,
    female_enrollment: Path,
) -> list[dict[str, Any]]:
    language_name = {"zh": "Chinese", "en": "English"}[language]
    enrollments = {
        "male": str(male_enrollment.resolve()),
        "female": str(female_enrollment.resolve()),
    }
    prepared: list[dict[str, Any]] = []
    for task in TASKS:
        for index in selections[task]:
            source = source_rows[index]
            source_id = str(source["id"])
            gender = infer_gender(str(source["APS"]), language)
            reference_path = output_dir / "reference_audio" / f"{source_id}.wav"
            extract_reference_audio(source["reference_audio"], reference_path)
            prepared.append(
                {
                    "sample_id": f"{source_id}_{task}",
                    "source_id": source_id,
                    "benchmark": "InstructTTSEval",
                    "task": task,
                    "text": str(source["text"]),
                    "instruction": str(source[task]),
                    "aps_instruction": str(source["APS"]),
                    "language": language_name,
                    "gender": gender,
                    "speaker_id": gender,
                    "scene_id": source_id,
                    "emotion": "",
                    "student_spk_audio": enrollments[gender],
                    "target_audio": str(reference_path.resolve()),
                    "reference_audio": str(reference_path.resolve()),
                }
            )
    return prepared


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare a leak-free voice-clone adaptation of InstructTTSEval."
    )
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--language", choices=["zh", "en"], required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--male-enrollment", type=Path, required=True)
    parser.add_argument("--female-enrollment", type=Path, required=True)
    parser.add_argument(
        "--num-per-task",
        type=int,
        default=20,
        help="Number of disjoint source rows per task; 0 expands all rows for every task.",
    )
    parser.add_argument("--seed", type=int, default=20260716)
    args = parser.parse_args()

    for enrollment in (args.male_enrollment, args.female_enrollment):
        if not enrollment.expanduser().is_file():
            parser.error(f"enrollment audio does not exist: {enrollment}")

    import pyarrow.parquet as pq

    parquet = args.parquet.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    source_rows = pq.read_table(parquet).to_pylist()
    selections = select_task_indices(len(source_rows), args.num_per_task, args.seed)
    rows = prepare_rows(
        source_rows,
        selections,
        language=args.language,
        output_dir=output_dir,
        male_enrollment=args.male_enrollment.expanduser(),
        female_enrollment=args.female_enrollment.expanduser(),
    )

    output_jsonl = output_dir / f"{args.language}_eval.jsonl"
    with output_jsonl.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "benchmark": "InstructTTSEval",
        "source_parquet": str(parquet),
        "language": args.language,
        "source_rows": len(source_rows),
        "num_per_task": args.num_per_task,
        "prepared_rows": len(rows),
        "task_counts": {task: len(selections[task]) for task in TASKS},
        "gender_counts": {
            gender: sum(row["gender"] == gender for row in rows)
            for gender in ("male", "female")
        },
        "male_enrollment": str(args.male_enrollment.expanduser().resolve()),
        "female_enrollment": str(args.female_enrollment.expanduser().resolve()),
        "seed": args.seed,
        "evaluation_note": (
            "Voice-clone adaptation: independent gender-matched EmotionTalk enrollment is shared "
            "by all models; benchmark reference audio is never used as model input."
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({**summary, "output_jsonl": str(output_jsonl)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
