from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable


TAG_RE = re.compile(r"\[/?[A-Za-z_]+/?\]")
NUMBER_RE = re.compile(r"_(\d+)$")
OFFICIAL_VAL_GROUPS = {"G00001", "G00012"}
OFFICIAL_TEST_GROUPS = {"G00003", "G00015"}


@dataclass(frozen=True)
class Utterance:
    sample_id: str
    key: str
    speaker_id: str
    scene_id: str
    sequence: int
    audio: Path
    text: str
    instruction: str
    emotion: str


def clean_transcript(text: str) -> str:
    return re.sub(r"\s+", " ", TAG_RE.sub("", text)).strip()


def parse_emotiontalk_key(value: str) -> tuple[str, str, int, str]:
    key = str(PurePosixPath(value)).removesuffix(".wav")
    parts = PurePosixPath(key).parts
    if len(parts) < 4:
        raise ValueError(f"expected EmotionTalk path with at least four components: {value}")
    speaker_match = NUMBER_RE.search(parts[-2])
    sequence_match = NUMBER_RE.search(parts[-1])
    if speaker_match is None or sequence_match is None:
        raise ValueError(f"cannot parse speaker/sequence suffix from: {value}")
    speaker_id = speaker_match.group(1)
    scene_id = "/".join(parts[:-2])
    sample_id = "__".join(parts)
    return speaker_id, scene_id, int(sequence_match.group(1)), sample_id


def parse_caption(raw: str, key: str) -> str:
    try:
        value = ast.literal_eval(raw)
    except (SyntaxError, ValueError) as exc:
        raise ValueError(f"invalid caption dictionary: {raw[:120]}") from exc
    if not isinstance(value, dict):
        raise ValueError("caption content must be a dictionary")
    if key == "combined":
        fields = [value.get(name) for name in ("spe_cap", "style_cap", "emo_cap")]
        caption = " ".join(str(field).strip() for field in fields if field)
    else:
        caption = str(value.get(key, "")).strip()
    if not caption:
        raise ValueError(f"caption field {key!r} is empty")
    return caption


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _audio_index(audio_root: Path) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = defaultdict(list)
    for path in audio_root.rglob("*.wav"):
        index[path.name].append(path.resolve())
    return index


def resolve_audio(audio_root: Path, relative: str, index: dict[str, list[Path]]) -> Path:
    rel = Path(PurePosixPath(relative))
    candidates = (
        audio_root / rel,
        audio_root / "Audio" / rel,
        audio_root / "audio" / rel,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    matches = index.get(rel.name, [])
    suffix = rel.as_posix()
    suffix_matches = [path for path in matches if path.as_posix().endswith(suffix)]
    if len(suffix_matches) == 1:
        return suffix_matches[0]
    if len(matches) == 1:
        return matches[0]
    return (audio_root / rel).resolve()


def split_for_scene(scene_id: str) -> str:
    group_id = PurePosixPath(scene_id).parts[0]
    if group_id in OFFICIAL_VAL_GROUPS:
        return "val"
    if group_id in OFFICIAL_TEST_GROUPS:
        return "test"
    return "train"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cached_sha256(path: Path, cache: dict[Path, str]) -> str:
    if path not in cache:
        cache[path] = file_sha256(path)
    return cache[path]


def load_utterances(
    transcription_csv: Path,
    caption_csv: Path,
    audio_root: Path,
    caption_key: str,
    check_audio: bool,
) -> list[Utterance]:
    transcripts = _read_csv(transcription_csv)
    captions = {row["file_name"].removesuffix(".wav"): row for row in _read_csv(caption_csv)}
    index = _audio_index(audio_root) if check_audio else {}
    utterances: list[Utterance] = []
    seen: set[str] = set()
    for row in transcripts:
        key = row.get("name", "").removesuffix(".wav")
        if not key or key in seen:
            raise ValueError(f"empty or duplicate transcription key: {key!r}")
        seen.add(key)
        caption_row = captions.get(key)
        if caption_row is None:
            raise ValueError(f"missing audio caption for {key}")
        speaker_id, scene_id, sequence, sample_id = parse_emotiontalk_key(key)
        audio = resolve_audio(audio_root, f"{key}.wav", index)
        if check_audio and not audio.is_file():
            raise FileNotFoundError(f"audio not found for {key}: {audio}")
        text = clean_transcript(row.get("chinese", ""))
        if not text:
            raise ValueError(f"empty transcript after tag cleaning: {key}")
        utterances.append(
            Utterance(
                sample_id=sample_id,
                key=key,
                speaker_id=speaker_id,
                scene_id=scene_id,
                sequence=sequence,
                audio=audio,
                text=text,
                instruction=parse_caption(caption_row.get("content", ""), caption_key),
                emotion=caption_row.get("emotion", row.get("emotion", "")),
            )
        )
    return utterances


def _base_row(item: Utterance, enrollment: Utterance | None) -> dict[str, Any]:
    row = {
        "sample_id": item.sample_id,
        "text": item.text,
        "instruction": item.instruction,
        "language": "Chinese",
        "speaker_id": item.speaker_id,
        "scene_id": item.scene_id,
        "sequence": item.sequence,
        "emotion": item.emotion,
        "audio": str(item.audio),
        "target_audio": str(item.audio),
    }
    if enrollment is not None:
        row["student_spk_audio"] = str(enrollment.audio)
        row["ref_audio"] = str(enrollment.audio)
    return row


def _group_problem(
    group: list[Utterance],
    check_audio_hash: bool,
    hash_cache: dict[Path, str],
) -> str | None:
    if len(group) < 2:
        return "single_utterance"
    numbers = [item.sequence for item in group]
    if len(numbers) != len(set(numbers)):
        return "duplicate_sequence"
    paths = [item.audio.resolve() for item in group]
    if len(paths) != len(set(paths)):
        return "duplicate_audio_path"
    texts = [item.text for item in group]
    for index, text in enumerate(texts):
        if text == texts[(index + 1) % len(texts)]:
            return "same_text_pair"
    if check_audio_hash:
        hashes = [cached_sha256(path, hash_cache) for path in paths]
        if len(hashes) != len(set(hashes)):
            return "duplicate_audio_content"
    return None


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def convert(
    utterances: list[Utterance],
    output_dir: Path,
    on_invalid_group: str,
    check_audio_hash: bool,
    limit: int | None,
    student_mode: str = "base",
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if student_mode not in {"base", "voice_design"}:
        raise ValueError(f"unsupported student_mode: {student_mode}")
    enrollment: dict[tuple[str, str], Utterance] = {}
    if student_mode == "base":
        by_speaker_split: dict[tuple[str, str], list[Utterance]] = defaultdict(list)
        for item in utterances:
            by_speaker_split[(split_for_scene(item.scene_id), item.speaker_id)].append(item)
        enrollment = {
            key: min(items, key=lambda item: (item.scene_id, item.sequence, item.key))
            for key, items in by_speaker_split.items()
        }
        enrollment_ids = {item.sample_id for item in enrollment.values()}
        targets = [item for item in utterances if item.sample_id not in enrollment_ids]
    else:
        targets = list(utterances)
    targets.sort(key=lambda item: (item.scene_id, item.speaker_id, item.sequence, item.key))
    if limit is not None:
        targets = targets[:limit]

    sft: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in targets:
        split = split_for_scene(item.scene_id)
        sft[split].append(_base_row(item, enrollment.get((split, item.speaker_id))))

    grouped: dict[tuple[str, str], list[Utterance]] = defaultdict(list)
    for item in targets:
        grouped[(item.speaker_id, item.scene_id)].append(item)

    opd: dict[str, list[dict[str, Any]]] = defaultdict(list)
    audit: list[dict[str, Any]] = []
    skipped = Counter()
    hash_cache: dict[Path, str] = {}
    for (speaker_id, scene_id), group in sorted(grouped.items()):
        group.sort(key=lambda item: (item.sequence, item.key))
        problem = _group_problem(group, check_audio_hash, hash_cache)
        if problem is None and check_audio_hash and student_mode == "base":
            split = split_for_scene(scene_id)
            enrollment_audio = enrollment[(split, speaker_id)].audio.resolve()
            enrollment_hash = cached_sha256(enrollment_audio, hash_cache)
            if any(cached_sha256(item.audio.resolve(), hash_cache) == enrollment_hash for item in group):
                problem = "target_matches_student_enrollment_content"
        if problem is not None:
            skipped[problem] += 1
            audit.append(
                {
                    "speaker_id": speaker_id,
                    "scene_id": scene_id,
                    "status": "skipped",
                    "reason": problem,
                    "sample_ids": [item.sample_id for item in group],
                }
            )
            if on_invalid_group == "error":
                raise ValueError(f"invalid group ({speaker_id}, {scene_id}): {problem}")
            continue
        cycle = []
        for index, item in enumerate(group):
            teacher = group[(index + 1) % len(group)]
            split = split_for_scene(item.scene_id)
            row = _base_row(item, enrollment.get((split, item.speaker_id)))
            row.update(
                {
                    "teacher_ref_audio": str(teacher.audio),
                    "teacher_ref_text": teacher.text,
                    "teacher_ref_sample_id": teacher.sample_id,
                    "ref_audio": str(teacher.audio),
                    "ref_text": teacher.text,
                }
            )
            opd[split].append(row)
            cycle.append(
                {
                    "sample_id": item.sample_id,
                    "teacher_ref_sample_id": teacher.sample_id,
                    "target_audio": str(item.audio),
                    "teacher_ref_audio": str(teacher.audio),
                }
            )
        audit.append(
            {
                "speaker_id": speaker_id,
                "scene_id": scene_id,
                "status": "ok",
                "size": len(group),
                "cycle": cycle,
            }
        )

    counts: dict[str, Any] = {
        "source_rows": len(utterances),
        "student_mode": student_mode,
        "enrollment_rows": len(enrollment),
    }
    for split in ("train", "val", "test"):
        target_counts = Counter(row["target_audio"] for row in opd[split])
        reference_counts = Counter(row["teacher_ref_audio"] for row in opd[split])
        if target_counts != reference_counts or any(count != 1 for count in reference_counts.values()):
            raise AssertionError(f"{split} OPD cycle does not use every target exactly once as teacher reference")
        for row in opd[split]:
            if row["target_audio"] == row["teacher_ref_audio"]:
                raise AssertionError(f"audio leakage in {row['sample_id']}")
            if student_mode == "base" and row["target_audio"] == row["student_spk_audio"]:
                raise AssertionError(f"student enrollment leakage in {row['sample_id']}")
        counts[f"sft_{split}"] = write_jsonl(output_dir / f"sft_{split}.jsonl", sft[split])
        counts[f"opd_{split}"] = write_jsonl(output_dir / f"opd_{split}.jsonl", opd[split])
    counts["audit_groups"] = write_jsonl(output_dir / "group_audit.jsonl", audit)
    counts["valid_groups"] = sum(row["status"] == "ok" for row in audit)
    counts["skipped_groups"] = dict(sorted(skipped.items()))
    counts["opd_reference_cycle_verified"] = True
    counts["target_teacher_audio_leaks"] = 0
    counts["target_student_audio_leaks"] = 0
    counts["license"] = "CC BY-NC-SA 4.0 (non-commercial)"
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(counts, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert EmotionTalk annotations into Qwen3-TTS SFT and OPD JSONL.")
    parser.add_argument("--transcription-csv", type=Path, required=True)
    parser.add_argument("--caption-csv", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--caption-key", default="caption_1", choices=["caption_1", "caption_2", "caption_3", "caption_4", "caption_5", "emo_cap", "spe_cap", "style_cap", "combined"])
    parser.add_argument("--on-invalid-group", choices=["error", "skip"], default="skip")
    parser.add_argument("--check-audio", action="store_true")
    parser.add_argument("--check-audio-hash", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--student-mode", choices=["base", "voice_design"], default="base")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    utterances = load_utterances(
        args.transcription_csv,
        args.caption_csv,
        args.audio_root.resolve(),
        args.caption_key,
        args.check_audio,
    )
    summary = convert(
        utterances,
        args.output_dir,
        args.on_invalid_group,
        args.check_audio_hash,
        args.limit,
        student_mode=args.student_mode,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
