from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Iterable


AUDIO_FIELDS = {
    "audio",
    "target_audio",
    "student_spk_audio",
    "ref_audio",
    "teacher_ref_audio",
}
DEFAULT_JSONL_FILES = (
    "sft_train.jsonl",
    "sft_val.jsonl",
    "sft_test.jsonl",
    "opd_train.jsonl",
    "opd_val.jsonl",
    "opd_test.jsonl",
    "group_audit.jsonl",
)


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            yield value


def _map_audio_paths(value: Any, transform: Callable[[str], str]) -> Any:
    if isinstance(value, list):
        return [_map_audio_paths(item, transform) for item in value]
    if isinstance(value, dict):
        mapped: dict[str, Any] = {}
        for key, item in value.items():
            if key in AUDIO_FIELDS and isinstance(item, str) and item:
                mapped[key] = transform(item)
            else:
                mapped[key] = _map_audio_paths(item, transform)
        return mapped
    return value


def _relative_transform(audio_root: Path) -> Callable[[str], str]:
    root = audio_root.expanduser().resolve()

    def transform(raw_path: str) -> str:
        path = Path(raw_path).expanduser()
        resolved = path.resolve() if path.is_absolute() else (root / path).resolve()
        try:
            return resolved.relative_to(root).as_posix()
        except ValueError as exc:
            raise ValueError(f"audio path is outside --audio-root: {raw_path}") from exc

    return transform


def _absolute_transform(audio_root: Path, check_audio: bool) -> Callable[[str], str]:
    root = audio_root.expanduser().resolve()

    def transform(raw_path: str) -> str:
        path = Path(raw_path)
        if path.is_absolute():
            raise ValueError(f"portable manifest contains an absolute path: {raw_path}")
        resolved = (root / path).resolve()
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"portable path escapes --audio-root: {raw_path}") from exc
        if check_audio and not resolved.is_file():
            raise FileNotFoundError(f"audio file not found: {resolved}")
        return str(resolved)

    return transform


def _write_jsonl(
    input_path: Path,
    output_path: Path,
    transform: Callable[[str], str],
    overwrite: bool,
) -> int:
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"output exists; pass --overwrite: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    count = 0
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for row in _iter_jsonl(input_path):
                mapped = _map_audio_paths(row, transform)
                handle.write(json.dumps(mapped, ensure_ascii=False, separators=(",", ":")) + "\n")
                count += 1
        temporary.replace(output_path)
    finally:
        temporary.unlink(missing_ok=True)
    return count


def convert_directory(
    input_dir: Path,
    output_dir: Path,
    audio_root: Path,
    mode: str,
    files: Iterable[str] = DEFAULT_JSONL_FILES,
    check_audio: bool = False,
    overwrite: bool = False,
) -> dict[str, int]:
    if mode == "export":
        transform = _relative_transform(audio_root)
    elif mode == "materialize":
        transform = _absolute_transform(audio_root, check_audio=check_audio)
    else:
        raise ValueError(f"unsupported mode: {mode}")

    counts: dict[str, int] = {}
    for filename in files:
        source = input_dir / filename
        if not source.is_file():
            raise FileNotFoundError(f"manifest not found: {source}")
        counts[filename] = _write_jsonl(
            source,
            output_dir / filename,
            transform,
            overwrite=overwrite,
        )

    summary_source = input_dir / "summary.json"
    if summary_source.is_file():
        summary_output = output_dir / "summary.json"
        if summary_output.exists() and not overwrite:
            raise FileExistsError(f"output exists; pass --overwrite: {summary_output}")
        summary_output.write_text(
            json.dumps(
                json.loads(summary_source.read_text(encoding="utf-8")),
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    return counts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export or materialize OPD manifests without machine-specific audio paths."
    )
    parser.add_argument("mode", choices=("export", "materialize"))
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument(
        "--files",
        nargs="+",
        default=list(DEFAULT_JSONL_FILES),
        help="JSONL filenames relative to --input-dir",
    )
    parser.add_argument("--check-audio", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.mode == "export" and args.check_audio:
        raise ValueError("--check-audio is only meaningful in materialize mode")
    counts = convert_directory(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        audio_root=args.audio_root,
        mode=args.mode,
        files=args.files,
        check_audio=args.check_audio,
        overwrite=args.overwrite,
    )
    print(json.dumps({"mode": args.mode, "files": counts}, indent=2))


if __name__ == "__main__":
    main()
