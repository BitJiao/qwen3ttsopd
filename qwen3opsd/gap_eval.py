from __future__ import annotations

import argparse
import html
import json
import os
import random
import time
from pathlib import Path
from typing import Any

from qwen3opsd.compare_vd_teacher import (
    generation_kwargs,
    read_jsonl,
    relative_path,
    sample_seed,
    set_seed,
    write_manifest,
)


HARD_CASES = (
    (
        "cold_anger",
        "G00015__G00015_14__G00015_14_21__G00015_14_21_013",
        "这不是第一次了。我不想提高声音，但你必须现在把事情解释清楚。",
    ),
    (
        "trembling_anger",
        "G00015__G00015_14__G00015_14_21__G00015_14_21_019",
        "你明明答应过我，为什么到了最后一刻又要反悔？",
    ),
    (
        "calm_sarcasm",
        "G00015__G00015_07__G00015_07_20__G00015_07_20_025",
        "原来这就是你所谓的专业，今天我可算见识到了。",
    ),
    (
        "anxious_fear",
        "G00015__G00015_18__G00015_18_20__G00015_18_20_013",
        "电话一直没人接，她从来不会这么晚还不回消息，肯定出事了。",
    ),
    (
        "panicked_resistance",
        "G00003__G00003_23__G00003_23_13__G00003_23_13_005",
        "等一下，别推我，我真的还没有准备好，让我先下来！",
    ),
    (
        "sobbing_sadness",
        "G00003__G00003_30__G00003_30_02__G00003_30_02_009",
        "我知道你迟早要走，可我还是希望今天能够过得慢一点。",
    ),
    (
        "soft_happiness",
        "G00015__G00015_11__G00015_11_20__G00015_11_20_001",
        "忙了这么久终于可以坐下来，好好跟你说说最近发生的事了。",
    ),
    (
        "respectful_excitement",
        "G00015__G00015_20__G00015_20_20__G00015_20_20_016",
        "真的太感谢您了，这个机会对我们全家来说都特别重要。",
    ),
    (
        "urgent_surprise",
        "G00003__G00003_11__G00003_11_13__G00003_11_13_013",
        "什么？消息已经公布了？怎么没有人提前告诉我！",
    ),
    (
        "raspy_irony",
        "G00015__G00015_07__G00015_07_20__G00015_07_20_006",
        "聊了半天，你对我一无所知，倒是把自己的大道理讲得很明白。",
    ),
    (
        "variable_speed_anger",
        "G00015__G00015_06__G00015_06_20__G00015_06_20_029",
        "我已经很克制了，但你一次又一次打断我，真的太不尊重人了。",
    ),
    (
        "loud_fast_anger",
        "G00015__G00015_09__G00015_09_20__G00015_09_20_015",
        "先把它拉住！这么多人在这里，你怎么还能不拴绳？",
    ),
)


def build_gap_cases(rows: list[dict[str, Any]], limit: int = 0) -> list[dict[str, Any]]:
    by_id = {str(row["sample_id"]): row for row in rows}
    selected = HARD_CASES[:limit] if limit > 0 else HARD_CASES
    cases: list[dict[str, Any]] = []
    for case_name, reference_id, target_text in selected:
        if reference_id not in by_id:
            raise KeyError(f"EmotionTalk reference is missing: {reference_id}")
        source = by_id[reference_id]
        reference_audio = source.get("target_audio", source.get("audio"))
        reference_text = str(source.get("text", "")).strip()
        caption = str(source.get("instruction", "")).strip()
        if not reference_audio or not Path(reference_audio).is_file():
            raise FileNotFoundError(f"reference audio is missing: {reference_audio}")
        if not caption or not reference_text:
            raise ValueError(f"reference text/caption is empty: {reference_id}")
        if target_text.strip() == reference_text:
            raise ValueError(f"target text leaks reference transcript: {reference_id}")
        cases.append(
            {
                "sample_id": case_name,
                "case_name": case_name,
                "emotion": str(source.get("emotion", "")),
                "text": target_text,
                "instruction": caption,
                "teacher_ref_audio": str(Path(reference_audio).resolve()),
                "teacher_ref_text": reference_text,
                "teacher_ref_sample_id": reference_id,
                "language": str(source.get("language", "Chinese")),
                "student_conditioning": "target_text + caption",
                "teacher_conditioning": "target_text + ref_audio + ref_transcript",
            }
        )
    return cases


def build_blind_rows(cases: list[dict[str, Any]], output_dir: Path, seed: int):
    rows: list[dict[str, Any]] = []
    key: dict[str, dict[str, str]] = {}
    for case in cases:
        systems = [
            {
                "model": "VD Student",
                "audio_src": relative_path(case["student_vd_audio"], output_dir),
            },
            {
                "model": "ICL Teacher",
                "audio_src": relative_path(case["teacher_icl_audio"], output_dir),
            },
        ]
        random.Random(f"{seed}:{case['sample_id']}").shuffle(systems)
        for index, system in enumerate(systems):
            system["system_id"] = chr(ord("A") + index)
        key[case["sample_id"]] = {
            system["system_id"]: system["model"] for system in systems
        }
        rows.append(
            {
                "sample_id": case["sample_id"],
                "emotion": case["emotion"],
                "text": case["text"],
                "instruction": case["instruction"],
                "reference_text": case["teacher_ref_text"],
                "reference_src": relative_path(case["teacher_ref_audio"], output_dir),
                "systems": systems,
            }
        )
    return rows, key


HTML_TEMPLATE = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>VD vs ICL Hard Gap</title>
  <style>
    :root { color-scheme:light; --ink:#181a1d; --muted:#626870; --line:#d8dde2; --bg:#f3f4f5; --surface:#fff; --accent:#176b45; --reveal:#8a4b12; }
    * { box-sizing:border-box; }
    body { margin:0; background:var(--bg); color:var(--ink); font:14px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif; letter-spacing:0; }
    header { position:sticky; top:0; z-index:3; border-bottom:1px solid var(--line); background:rgba(255,255,255,.97); }
    .toolbar,main { width:min(1220px,calc(100% - 32px)); margin:auto; }
    .toolbar { padding:12px 0; display:grid; grid-template-columns:1fr auto auto; gap:12px; align-items:center; }
    h1 { margin:0; font-size:18px; } .summary,.meta { color:var(--muted); font-size:12px; }
    button { min-height:34px; padding:6px 10px; border:1px solid #b9c0c7; border-radius:4px; background:#fff; color:var(--ink); font:inherit; cursor:pointer; }
    main { padding:16px 0 60px; }
    article { margin-bottom:14px; border:1px solid var(--line); border-radius:6px; background:var(--surface); overflow:hidden; }
    .head { padding:12px 14px; border-bottom:1px solid var(--line); display:grid; grid-template-columns:130px minmax(220px,.85fr) minmax(320px,1.3fr); gap:15px; }
    .label { display:block; margin-bottom:3px; color:var(--muted); font-size:11px; font-weight:700; text-transform:uppercase; }
    .case { font:12px ui-monospace,SFMono-Regular,Consolas,monospace; overflow-wrap:anywhere; }
    .reference { padding:10px 14px; border-bottom:1px solid var(--line); display:grid; grid-template-columns:1fr 340px; gap:15px; align-items:center; background:#fafbfb; }
    .systems { display:grid; grid-template-columns:1fr 1fr; }
    .system { min-width:0; padding:12px 14px 14px; }
    .system + .system { border-left:1px solid var(--line); }
    .system-title { display:flex; justify-content:space-between; gap:8px; margin-bottom:8px; font-size:15px; font-weight:700; }
    .model { color:var(--reveal); font-size:12px; }
    audio { display:block; width:100%; height:36px; }
    .choice { padding:10px 14px; border-top:1px solid var(--line); display:flex; align-items:center; gap:16px; flex-wrap:wrap; }
    .choice label { cursor:pointer; } .choice strong { color:var(--accent); }
    @media(max-width:780px) { .toolbar,.head,.reference,.systems { grid-template-columns:1fr; } .system + .system { border-left:0; border-top:1px solid var(--line); } }
  </style>
</head>
<body>
<header><div class="toolbar"><div><h1>VD vs ICL Hard Gap</h1><div class="summary" id="summary"></div></div><button id="reveal-all" type="button">Reveal all</button><button id="export" type="button">Export ratings</button></div></header>
<main id="root"></main>
<script>
const ROWS=__ROWS__;
const STORE="vd-icl-hard-gap-v1";
let ratings=JSON.parse(localStorage.getItem(STORE)||"{}");
const esc=value=>String(value).replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
function save(id,value){ratings[id]=value;localStorage.setItem(STORE,JSON.stringify(ratings));summary();}
function summary(){const count=Object.values(ratings).filter(Boolean).length;document.getElementById("summary").textContent=`${ROWS.length} cases · ${count} rated · listen to the reference first`;}
function render(){document.getElementById("root").innerHTML=ROWS.map((row,index)=>{
  const systems=row.systems.map(system=>`<section class="system"><div class="system-title"><span>System ${system.system_id}</span><span class="model" hidden>${esc(system.model)}</span></div><audio controls preload="metadata" src="${esc(system.audio_src)}"></audio></section>`).join("");
  const choices=row.systems.map(system=>`<label><input type="radio" name="choice-${index}" value="${system.system_id}" ${ratings[row.sample_id]===system.system_id?"checked":""}> System ${system.system_id}</label>`).join("");
  return `<article data-id="${esc(row.sample_id)}"><div class="head"><div><span class="label">Case / emotion</span><div class="case">${esc(row.sample_id)}</div><div>${esc(row.emotion)}</div></div><div><span class="label">Target text</span>${esc(row.text)}</div><div><span class="label">Caption (student only)</span>${esc(row.instruction)}</div></div><div class="reference"><div><span class="label">ICL reference transcript</span>${esc(row.reference_text)}</div><audio controls preload="metadata" src="${esc(row.reference_src)}"></audio></div><div class="systems">${systems}</div><div class="choice"><strong>Closer to reference style:</strong>${choices}<label><input type="radio" name="choice-${index}" value="tie" ${ratings[row.sample_id]==="tie"?"checked":""}> Tie</label><button class="reveal" type="button">Reveal</button></div></article>`;
}).join("");
document.querySelectorAll('.choice input').forEach(input=>input.addEventListener('change',event=>save(event.target.closest('article').dataset.id,event.target.value)));
document.querySelectorAll('.reveal').forEach(button=>button.addEventListener('click',event=>{const labels=event.target.closest('article').querySelectorAll('.model');const show=[...labels].some(label=>label.hidden);labels.forEach(label=>label.hidden=!show);event.target.textContent=show?'Hide':'Reveal';}));summary();}
document.getElementById('reveal-all').addEventListener('click',()=>document.querySelectorAll('.model').forEach(label=>label.hidden=false));
document.getElementById('export').addEventListener('click',()=>{const blob=new Blob([JSON.stringify({ratings,rows:ROWS},null,2)],{type:'application/json'});const link=document.createElement('a');link.href=URL.createObjectURL(blob);link.download='vd_icl_gap_ratings.json';link.click();URL.revokeObjectURL(link.href);});
render();
</script>
</body>
</html>
"""


def write_gap_report(output_dir: Path, cases: list[dict[str, Any]], seed: int) -> None:
    rows, key = build_blind_rows(cases, output_dir, seed)
    payload = json.dumps(rows, ensure_ascii=False).replace("<", "\\u003c")
    (output_dir / "listen.html").write_text(
        HTML_TEMPLATE.replace("__ROWS__", payload), encoding="utf-8"
    )
    (output_dir / "blind_key.json").write_text(
        json.dumps(key, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a blind hard-caption gap test for VD and ICL.")
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--student-model-path", type=Path, required=True)
    parser.add_argument("--teacher-model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--student-device", default="cuda:1")
    parser.add_argument("--teacher-device", default="cuda:2")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    if args.limit < 0 or args.limit > len(HARD_CASES):
        parser.error(f"--limit must be between 0 and {len(HARD_CASES)}")
    output_dir = args.output_dir.expanduser().resolve()
    student_dir = output_dir / "audio" / "student_vd"
    teacher_dir = output_dir / "audio" / "teacher_icl"
    student_dir.mkdir(parents=True, exist_ok=True)
    teacher_dir.mkdir(parents=True, exist_ok=True)
    cases = build_gap_cases(read_jsonl(args.input_jsonl.expanduser().resolve()), args.limit)

    import numpy as np
    import soundfile as sf
    import torch
    from qwen_tts import Qwen3TTSModel

    dtype = getattr(torch, args.dtype)
    load_kwargs = {"dtype": dtype, "attn_implementation": args.attn_implementation}
    kwargs = generation_kwargs(args)
    student = Qwen3TTSModel.from_pretrained(
        str(args.student_model_path.expanduser().resolve()),
        device_map=args.student_device,
        **load_kwargs,
    )
    if student.model.tts_model_type != "voice_design":
        raise ValueError("student must be a VoiceDesign checkpoint")
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
        output = student_dir / f"{index:02d}_{case['sample_id']}.wav"
        sf.write(output, wavs[0], sample_rate)
        case["student_vd_audio"] = str(output.resolve())
        case["student_seed"] = seed
        print(json.dumps({"system": "student_vd", "case": index, "name": case["sample_id"], "seconds": round(time.monotonic() - started, 2)}, ensure_ascii=False), flush=True)
    del student
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    teacher = Qwen3TTSModel.from_pretrained(
        str(args.teacher_model_path.expanduser().resolve()),
        device_map=args.teacher_device,
        **load_kwargs,
    )
    if teacher.model.tts_model_type != "base":
        raise ValueError("teacher must be a Base checkpoint")
    for index, case in enumerate(cases, 1):
        started = time.monotonic()
        seed = sample_seed(args.seed, case["sample_id"])
        set_seed(seed, torch, np)
        wavs, sample_rate = teacher.generate_voice_clone(
            text=case["text"],
            language=case["language"],
            ref_audio=case["teacher_ref_audio"],
            ref_text=case["teacher_ref_text"],
            x_vector_only_mode=False,
            non_streaming_mode=True,
            **kwargs,
        )
        output = teacher_dir / f"{index:02d}_{case['sample_id']}.wav"
        sf.write(output, wavs[0], sample_rate)
        case["teacher_icl_audio"] = str(output.resolve())
        case["teacher_seed"] = seed
        print(json.dumps({"system": "teacher_icl", "case": index, "name": case["sample_id"], "seconds": round(time.monotonic() - started, 2)}, ensure_ascii=False), flush=True)

    write_manifest(output_dir / "manifest.jsonl", cases)
    write_gap_report(output_dir, cases, args.seed)
    print(json.dumps({"cases": len(cases), "listen_html": str(output_dir / "listen.html"), "blind_key": str(output_dir / "blind_key.json")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
