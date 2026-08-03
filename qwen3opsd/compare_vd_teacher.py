from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path
from typing import Any

import torch


DEFAULT_CASES = (
    (
        "angry",
        "G00003__G00003_16__G00003_16_02__G00003_16_02_014",
        "我已经说过很多次了，这件事不能再拖，你到底明不明白？",
    ),
    (
        "happy",
        "G00003__G00003_23__G00003_23_02__G00003_23_02_019",
        "太好了，今天终于等到这个消息，我真的特别开心！",
    ),
    (
        "sad",
        "G00003__G00003_30__G00003_30_02__G00003_30_02_009",
        "我本来以为还有机会，没想到最后还是没能等到那一天。",
    ),
    (
        "surprised",
        "G00003__G00003_11__G00003_11_13__G00003_11_13_009",
        "什么？你说这件事已经解决了？这怎么可能！",
    ),
    (
        "fearful",
        "G00003__G00003_23__G00003_23_13__G00003_23_13_005",
        "外面好像有人，你先别出声，我去看看门有没有锁好。",
    ),
    (
        "disgusted",
        "G00003__G00003_17__G00003_17_13__G00003_17_13_016",
        "算了吧，这种做法我实在接受不了，也不想再谈了。",
    ),
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def build_demo_cases(rows: list[dict[str, Any]], limit: int = 0) -> list[dict[str, Any]]:
    by_id = {str(row["sample_id"]): row for row in rows}
    selected = DEFAULT_CASES[:limit] if limit > 0 else DEFAULT_CASES
    cases: list[dict[str, Any]] = []
    for emotion, reference_id, target_text in selected:
        if reference_id not in by_id:
            raise KeyError(f"EmotionTalk reference is missing: {reference_id}")
        source = by_id[reference_id]
        reference_audio = source.get("target_audio", source.get("audio"))
        reference_text = str(source.get("text", "")).strip()
        caption = str(source.get("instruction", "")).strip()
        if not reference_audio or not Path(reference_audio).is_file():
            raise FileNotFoundError(f"reference audio is missing: {reference_audio}")
        if not reference_text or not caption:
            raise ValueError(f"reference text/caption is empty: {reference_id}")
        if reference_text == target_text.strip():
            raise ValueError(f"reference and target text must differ: {reference_id}")
        cases.append(
            {
                "sample_id": f"{emotion}_{reference_id}",
                "emotion": emotion,
                "text": target_text,
                "instruction": caption,
                "teacher_ref_audio": str(Path(reference_audio).resolve()),
                "teacher_ref_text": reference_text,
                "teacher_ref_sample_id": reference_id,
                "language": str(source.get("language", "Chinese")),
            }
        )
    return cases


def sample_seed(base_seed: int, sample_id: str) -> int:
    digest = hashlib.sha256(sample_id.encode("utf-8")).digest()
    return (base_seed + int.from_bytes(digest[:4], "big")) % (2**31)


def set_seed(seed: int, torch_module, numpy_module) -> None:
    random.seed(seed)
    numpy_module.random.seed(seed)
    torch_module.manual_seed(seed)
    if torch_module.cuda.is_available():
        torch_module.cuda.manual_seed_all(seed)


def relative_path(path: str | Path, output_dir: Path) -> str:
    return Path(os.path.relpath(Path(path).resolve(), output_dir.resolve())).as_posix()


def write_manifest(path: Path, cases: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(json.dumps(case, ensure_ascii=False) + "\n")
    temporary.replace(path)


HTML_TEMPLATE = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>VD Student vs ICL Teacher</title>
  <style>
    :root { color-scheme: light; --ink:#17191c; --muted:#626870; --line:#d8dde2; --bg:#f4f5f6; --surface:#fff; --student:#9b3f28; --teacher:#176b45; }
    * { box-sizing:border-box; }
    body { margin:0; background:var(--bg); color:var(--ink); font:14px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif; letter-spacing:0; }
    header { position:sticky; top:0; z-index:2; border-bottom:1px solid var(--line); background:rgba(255,255,255,.96); }
    header div, main { width:min(1180px,calc(100% - 32px)); margin:auto; }
    header div { padding:14px 0; display:flex; align-items:baseline; justify-content:space-between; gap:16px; }
    h1 { margin:0; font-size:19px; }
    .summary,.meta { color:var(--muted); font-size:12px; }
    main { padding:18px 0 56px; }
    article { margin-bottom:14px; border:1px solid var(--line); border-radius:6px; background:var(--surface); overflow:hidden; }
    .case-head { padding:12px 14px; border-bottom:1px solid var(--line); display:grid; grid-template-columns:100px minmax(220px,.8fr) minmax(300px,1.2fr); gap:16px; }
    .label { display:block; margin-bottom:3px; color:var(--muted); font-size:11px; font-weight:700; text-transform:uppercase; }
    .emotion { font-weight:700; }
    .reference { padding:10px 14px; border-bottom:1px solid var(--line); display:grid; grid-template-columns:1fr 320px; gap:16px; align-items:center; }
    .reference audio { width:100%; }
    .systems { display:grid; grid-template-columns:1fr 1fr; }
    .system { padding:13px 14px 15px; min-width:0; }
    .system + .system { border-left:1px solid var(--line); }
    .system h2 { margin:0 0 9px; font-size:15px; }
    .student h2 { color:var(--student); } .teacher h2 { color:var(--teacher); }
    audio { display:block; width:100%; height:36px; }
    @media (max-width:760px) {
      .case-head,.reference,.systems { grid-template-columns:1fr; }
      .system + .system { border-left:0; border-top:1px solid var(--line); }
    }
  </style>
</head>
<body>
<header><div><h1>VD Student vs ICL Teacher</h1><span class="summary">EmotionTalk paired references</span></div></header>
<main>__CASES__</main>
</body>
</html>
"""


def write_html(path: Path, cases: list[dict[str, Any]], output_dir: Path) -> None:
    import html

    blocks = []
    for case in cases:
        student = case.get("student_vd_audio")
        teacher = case.get("teacher_icl_audio")
        if not student or not teacher:
            continue
        blocks.append(
            f"""<article>
  <div class="case-head">
    <div><span class="label">Emotion</span><span class="emotion">{html.escape(case['emotion'])}</span></div>
    <div><span class="label">Target text (same for both)</span>{html.escape(case['text'])}</div>
    <div><span class="label">Caption (same for both)</span>{html.escape(case['instruction'])}</div>
  </div>
  <div class="reference">
    <div><span class="label">Teacher reference transcript (different from target)</span>{html.escape(case['teacher_ref_text'])}<div class="meta">{html.escape(case['teacher_ref_sample_id'])}</div></div>
    <audio controls preload="metadata" src="{html.escape(relative_path(case['teacher_ref_audio'], output_dir))}"></audio>
  </div>
  <div class="systems">
    <section class="system student"><h2>Student: VoiceDesign (caption only)</h2><audio controls preload="metadata" src="{html.escape(relative_path(student, output_dir))}"></audio></section>
    <section class="system teacher"><h2>Teacher: Base ICL (caption + reference)</h2><audio controls preload="metadata" src="{html.escape(relative_path(teacher, output_dir))}"></audio></section>
  </div>
</article>"""
        )
    path.write_text(HTML_TEMPLATE.replace("__CASES__", "\n".join(blocks)), encoding="utf-8")


def generation_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "do_sample": True,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": 1.0,
        "subtalker_dosample": True,
        "subtalker_temperature": args.temperature,
        "subtalker_top_k": args.top_k,
        "subtalker_top_p": 1.0,
        "max_new_tokens": args.max_new_tokens,
    }


def generate_icl_teacher(
    teacher,
    *,
    text: str,
    instruction: str,
    reference_text: str,
    language: str,
    voice_clone_prompt: dict[str, Any],
    non_streaming_mode: bool = False,
    **generate_kwargs,
):
    input_id = teacher._tokenize_texts([teacher._build_assistant_text(text)])[0]
    instruct_id = teacher._tokenize_texts([teacher._build_instruct_text(instruction)])[0]
    reference_id = teacher._tokenize_texts([teacher._build_ref_text(reference_text)])[0]
    merged_kwargs = teacher._merge_generate_kwargs(**generate_kwargs)
    codes, _ = teacher.model.generate(
        input_ids=[input_id.to(teacher.device)],
        instruct_ids=[instruct_id.to(teacher.device)],
        ref_ids=[reference_id.to(teacher.device)],
        voice_clone_prompt=voice_clone_prompt,
        languages=[language],
        non_streaming_mode=non_streaming_mode,
        **merged_kwargs,
    )
    ref_code = voice_clone_prompt["ref_code"][0]
    codes_for_decode = torch.cat([ref_code.to(codes[0].device), codes[0]], dim=0)
    wavs, sample_rate = teacher.model.speech_tokenizer.decode(
        [{"audio_codes": codes_for_decode}]
    )
    ref_length = int(ref_code.shape[0])
    total_length = int(codes_for_decode.shape[0])
    cut = int(ref_length / max(total_length, 1) * wavs[0].shape[0])
    return [wavs[0][cut:]], sample_rate


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare a caption-only VoiceDesign student with a caption-matched ICL teacher."
    )
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--student-model-path", type=Path, required=True)
    parser.add_argument("--teacher-model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--student-device", default="cuda:1")
    parser.add_argument("--teacher-device", default="cuda:2")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--limit", type=int, default=0, help="0 runs all six predefined cases.")
    args = parser.parse_args()

    if args.limit < 0 or args.limit > len(DEFAULT_CASES):
        parser.error(f"--limit must be between 0 and {len(DEFAULT_CASES)}")
    output_dir = args.output_dir.expanduser().resolve()
    student_audio_dir = output_dir / "audio" / "student_vd"
    teacher_audio_dir = output_dir / "audio" / "teacher_icl"
    student_audio_dir.mkdir(parents=True, exist_ok=True)
    teacher_audio_dir.mkdir(parents=True, exist_ok=True)
    cases = build_demo_cases(read_jsonl(args.input_jsonl.expanduser().resolve()), args.limit)

    import numpy as np
    import soundfile as sf
    import torch
    from qwen_tts import Qwen3TTSModel

    dtype = getattr(torch, args.dtype)
    common_load = {"dtype": dtype, "attn_implementation": args.attn_implementation}
    student = Qwen3TTSModel.from_pretrained(
        str(args.student_model_path.expanduser().resolve()),
        device_map=args.student_device,
        **common_load,
    )
    if student.model.tts_model_type != "voice_design":
        raise ValueError("student must be a Qwen3-TTS VoiceDesign checkpoint")

    kwargs = generation_kwargs(args)
    for index, case in enumerate(cases, 1):
        started = time.monotonic()
        seed = sample_seed(args.seed, case["sample_id"])
        set_seed(seed, torch, np)
        wavs, sample_rate = student.generate_voice_design(
            text=case["text"],
            instruct=case["instruction"],
            language=case["language"],
            non_streaming_mode=True,
            **kwargs,
        )
        path = student_audio_dir / f"{case['emotion']}.wav"
        sf.write(path, wavs[0], sample_rate)
        case["student_vd_audio"] = str(path.resolve())
        case["student_seed"] = seed
        print(json.dumps({"system": "student_vd", "case": index, "emotion": case["emotion"], "seconds": round(time.monotonic() - started, 2)}, ensure_ascii=False), flush=True)

    del student
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    teacher = Qwen3TTSModel.from_pretrained(
        str(args.teacher_model_path.expanduser().resolve()),
        device_map=args.teacher_device,
        **common_load,
    )
    if teacher.model.tts_model_type != "base":
        raise ValueError("teacher must be a Qwen3-TTS Base checkpoint")

    for index, case in enumerate(cases, 1):
        started = time.monotonic()
        prompt_items = teacher.create_voice_clone_prompt(
            ref_audio=case["teacher_ref_audio"],
            ref_text=case["teacher_ref_text"],
            x_vector_only_mode=False,
        )
        prompt = teacher._prompt_items_to_voice_clone_prompt(prompt_items)
        seed = sample_seed(args.seed, case["sample_id"])
        set_seed(seed, torch, np)
        wavs, sample_rate = generate_icl_teacher(
            teacher,
            text=case["text"],
            instruction=case["instruction"],
            reference_text=case["teacher_ref_text"],
            language=case["language"],
            voice_clone_prompt=prompt,
            **kwargs,
        )
        path = teacher_audio_dir / f"{case['emotion']}.wav"
        sf.write(path, wavs[0], sample_rate)
        case["teacher_icl_audio"] = str(path.resolve())
        case["teacher_seed"] = seed
        print(json.dumps({"system": "teacher_icl", "case": index, "emotion": case["emotion"], "seconds": round(time.monotonic() - started, 2)}, ensure_ascii=False), flush=True)

    write_manifest(output_dir / "manifest.jsonl", cases)
    write_html(output_dir / "listen.html", cases, output_dir)
    print(json.dumps({"cases": len(cases), "listen_html": str(output_dir / "listen.html"), "manifest": str(output_dir / "manifest.jsonl")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
