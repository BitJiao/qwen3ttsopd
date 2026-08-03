from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from qwen3opsd.compare_vd_teacher import (
    generate_icl_teacher,
    generation_kwargs,
    read_jsonl,
    relative_path,
    sample_seed,
    set_seed,
    write_manifest,
)


def validate_cases(cases: list[dict[str, Any]]) -> None:
    if not cases:
        raise ValueError("comparison manifest has no cases")
    required = (
        "sample_id",
        "text",
        "instruction",
        "teacher_ref_audio",
        "teacher_ref_text",
        "student_vd_audio",
        "sft_vd_audio",
        "teacher_icl_audio",
    )
    for case in cases:
        missing = [field for field in required if not case.get(field)]
        if missing:
            raise ValueError(
                f"case {case.get('sample_id', '<unknown>')} is missing {missing}"
            )
        for field in (
            "teacher_ref_audio",
            "student_vd_audio",
            "sft_vd_audio",
            "teacher_icl_audio",
        ):
            if not Path(case[field]).is_file():
                raise FileNotFoundError(f"{field} does not exist: {case[field]}")


def build_named_rows(
    cases: list[dict[str, Any]], output_dir: Path
) -> list[dict[str, Any]]:
    rows = []
    for case in cases:
        rows.append(
            {
                "sample_id": case["sample_id"],
                "emotion": case.get("emotion", ""),
                "text": case["text"],
                "instruction": case["instruction"],
                "reference_text": case["teacher_ref_text"],
                "reference_src": relative_path(case["teacher_ref_audio"], output_dir),
                "systems": [
                    {
                        "model": "Original VoiceDesign",
                        "conditioning": "target text + caption",
                        "audio_src": relative_path(case["student_vd_audio"], output_dir),
                    },
                    {
                        "model": "SFT VoiceDesign",
                        "conditioning": "target text + caption",
                        "audio_src": relative_path(case["sft_vd_audio"], output_dir),
                    },
                    {
                        "model": "Base ICL",
                        "conditioning": "target text + reference audio + reference transcript",
                        "audio_src": relative_path(case["teacher_icl_audio"], output_dir),
                    },
                    {
                        "model": "Base ICL + Caption",
                        "conditioning": "target text + caption + reference audio + reference transcript",
                        "audio_src": relative_path(case["teacher_icl_caption_audio"], output_dir),
                    },
                ],
            }
        )
    return rows


HTML_TEMPLATE = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>VD / SFT VD / Base ICL / Base ICL + Caption</title>
  <style>
    :root { color-scheme:light; --ink:#17191c; --muted:#626870; --line:#d8dde2; --bg:#f3f4f5; --surface:#fff; --accent:#176b45; }
    * { box-sizing:border-box; }
    body { margin:0; background:var(--bg); color:var(--ink); font:14px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif; letter-spacing:0; }
    header { position:sticky; top:0; z-index:3; border-bottom:1px solid var(--line); background:rgba(255,255,255,.97); }
    header div,main { width:min(1500px,calc(100% - 32px)); margin:auto; }
    header div { padding:12px 0; display:flex; justify-content:space-between; align-items:baseline; gap:16px; }
    h1 { margin:0; font-size:18px; } .summary,.meta { color:var(--muted); font-size:12px; }
    main { padding:16px 0 60px; }
    article { margin-bottom:14px; border:1px solid var(--line); border-radius:6px; background:var(--surface); overflow:hidden; }
    .head { padding:12px 14px; border-bottom:1px solid var(--line); display:grid; grid-template-columns:130px minmax(220px,.85fr) minmax(320px,1.3fr); gap:15px; }
    .label { display:block; margin-bottom:3px; color:var(--muted); font-size:11px; font-weight:700; text-transform:uppercase; }
    .case { font:12px ui-monospace,SFMono-Regular,Consolas,monospace; overflow-wrap:anywhere; }
    .reference { padding:10px 14px; border-bottom:1px solid var(--line); display:grid; grid-template-columns:1fr 360px; gap:15px; align-items:center; background:#fafbfb; }
    .systems { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); }
    .system { min-width:0; padding:12px 14px 14px; }
    .system + .system { border-left:1px solid var(--line); }
    .system h2 { margin:0 0 2px; font-size:15px; color:var(--accent); }
    .conditioning { min-height:38px; margin-bottom:7px; color:var(--muted); font-size:12px; overflow-wrap:anywhere; }
    audio { display:block; width:100%; height:36px; }
    @media(max-width:1100px) { .systems { grid-template-columns:repeat(2,minmax(0,1fr)); } .system:nth-child(3) { border-left:0; border-top:1px solid var(--line); } .system:nth-child(4) { border-top:1px solid var(--line); } }
    @media(max-width:680px) { header div,.head,.reference,.systems { grid-template-columns:1fr; display:grid; } .system + .system { border-left:0; border-top:1px solid var(--line); } }
  </style>
</head>
<body>
<header><div><h1>VD / SFT VD / Base ICL / Base ICL + Caption</h1><span class="summary">__COUNT__ cases · named comparison</span></div></header>
<main id="root"></main>
<script>
const ROWS=__ROWS__;
const esc=value=>String(value).replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
document.getElementById("root").innerHTML=ROWS.map(row=>{
  const systems=row.systems.map(system=>`<section class="system"><h2>${esc(system.model)}</h2><div class="conditioning">${esc(system.conditioning)}</div><audio controls preload="metadata" src="${esc(system.audio_src)}"></audio></section>`).join("");
  return `<article><div class="head"><div><span class="label">Case / emotion</span><div class="case">${esc(row.sample_id)}</div><div>${esc(row.emotion)}</div></div><div><span class="label">Target text</span>${esc(row.text)}</div><div><span class="label">Caption</span>${esc(row.instruction)}</div></div><div class="reference"><div><span class="label">Reference transcript（reference 音频中实际说的话）</span>${esc(row.reference_text)}</div><audio controls preload="metadata" src="${esc(row.reference_src)}"></audio></div><div class="systems">${systems}</div></article>`;
}).join("");
</script>
</body>
</html>
"""


def write_report(output_dir: Path, cases: list[dict[str, Any]]) -> None:
    rows = build_named_rows(cases, output_dir)
    payload = json.dumps(rows, ensure_ascii=False).replace("<", "\\u003c")
    (output_dir / "listen.html").write_text(
        HTML_TEMPLATE.replace("__ROWS__", payload).replace("__COUNT__", str(len(rows))),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate caption+ICL audio and build a named four-way comparison."
    )
    parser.add_argument("--comparison-manifest", type=Path, required=True)
    parser.add_argument("--teacher-model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    cases = read_jsonl(args.comparison_manifest.expanduser().resolve())
    if args.limit < 0 or args.limit > len(cases):
        parser.error(f"--limit must be between 0 and {len(cases)}")
    if args.limit:
        cases = cases[: args.limit]
    validate_cases(cases)
    output_dir = args.output_dir.expanduser().resolve()
    audio_dir = output_dir / "audio" / "base_icl_caption"
    audio_dir.mkdir(parents=True, exist_ok=True)

    import numpy as np
    import soundfile as sf
    import torch
    from qwen_tts import Qwen3TTSModel

    teacher = Qwen3TTSModel.from_pretrained(
        str(args.teacher_model_path.expanduser().resolve()),
        device_map=args.device,
        dtype=getattr(torch, args.dtype),
        attn_implementation=args.attn_implementation,
    )
    if teacher.model.tts_model_type != "base":
        raise ValueError("teacher must be a Qwen3-TTS Base checkpoint")
    kwargs = generation_kwargs(args)
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
            language=case.get("language", "Chinese"),
            voice_clone_prompt=prompt,
            **kwargs,
        )
        output = audio_dir / f"{index:02d}_{case['sample_id']}.wav"
        sf.write(output, wavs[0], sample_rate)
        case["teacher_icl_caption_audio"] = str(output.resolve())
        case["teacher_icl_caption_seed"] = seed
        case["teacher_icl_caption_conditioning"] = (
            "target_text + caption + ref_audio + ref_transcript"
        )
        print(
            json.dumps(
                {
                    "system": "base_icl_caption",
                    "case": index,
                    "name": case["sample_id"],
                    "seconds": round(time.monotonic() - started, 2),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    write_manifest(output_dir / "manifest.jsonl", cases)
    write_report(output_dir, cases)
    print(
        json.dumps(
            {
                "cases": len(cases),
                "listen_html": str(output_dir / "listen.html"),
                "manifest": str(output_dir / "manifest.jsonl"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
