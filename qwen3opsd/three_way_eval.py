from __future__ import annotations

import argparse
import html
import json
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


def validate_baseline_cases(cases: list[dict[str, Any]]) -> None:
    if not cases:
        raise ValueError("baseline manifest has no cases")
    required = (
        "sample_id",
        "text",
        "instruction",
        "teacher_ref_audio",
        "teacher_ref_text",
        "student_vd_audio",
        "teacher_icl_audio",
    )
    for case in cases:
        missing = [name for name in required if not case.get(name)]
        if missing:
            raise ValueError(
                f"baseline case {case.get('sample_id', '<unknown>')} is missing {missing}"
            )
        for field in ("teacher_ref_audio", "student_vd_audio", "teacher_icl_audio"):
            if not Path(case[field]).is_file():
                raise FileNotFoundError(f"{field} does not exist: {case[field]}")


def build_blind_rows(
    cases: list[dict[str, Any]], output_dir: Path, seed: int
) -> tuple[list[dict[str, Any]], dict[str, dict[str, str]]]:
    rows: list[dict[str, Any]] = []
    key: dict[str, dict[str, str]] = {}
    for case in cases:
        systems = [
            {"model": "Original VoiceDesign", "audio_src": relative_path(case["student_vd_audio"], output_dir)},
            {"model": "SFT VoiceDesign", "audio_src": relative_path(case["sft_vd_audio"], output_dir)},
            {"model": "Base ICL", "audio_src": relative_path(case["teacher_icl_audio"], output_dir)},
        ]
        random.Random(f"three-way:{seed}:{case['sample_id']}").shuffle(systems)
        for index, system in enumerate(systems):
            system["system_id"] = chr(ord("A") + index)
        key[case["sample_id"]] = {
            system["system_id"]: system["model"] for system in systems
        }
        rows.append(
            {
                "sample_id": case["sample_id"],
                "emotion": case.get("emotion", ""),
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
  <title>Original VD / SFT VD / Base ICL</title>
  <style>
    :root { color-scheme:light; --ink:#17191c; --muted:#626870; --line:#d8dde2; --bg:#f3f4f5; --surface:#fff; --accent:#176b45; --reveal:#8a4b12; }
    * { box-sizing:border-box; }
    body { margin:0; background:var(--bg); color:var(--ink); font:14px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif; letter-spacing:0; }
    header { position:sticky; top:0; z-index:3; border-bottom:1px solid var(--line); background:rgba(255,255,255,.97); }
    .toolbar,main { width:min(1380px,calc(100% - 32px)); margin:auto; }
    .toolbar { padding:11px 0; display:grid; grid-template-columns:1fr auto auto; gap:10px; align-items:center; }
    h1 { margin:0; font-size:18px; } .summary,.meta { color:var(--muted); font-size:12px; }
    button { min-height:34px; padding:6px 10px; border:1px solid #b9c0c7; border-radius:4px; background:#fff; color:var(--ink); font:inherit; cursor:pointer; }
    main { padding:16px 0 60px; }
    article { margin-bottom:14px; border:1px solid var(--line); border-radius:6px; background:var(--surface); overflow:hidden; }
    .head { padding:12px 14px; border-bottom:1px solid var(--line); display:grid; grid-template-columns:130px minmax(220px,.85fr) minmax(320px,1.3fr); gap:15px; }
    .label { display:block; margin-bottom:3px; color:var(--muted); font-size:11px; font-weight:700; text-transform:uppercase; }
    .case { font:12px ui-monospace,SFMono-Regular,Consolas,monospace; overflow-wrap:anywhere; }
    .reference { padding:10px 14px; border-bottom:1px solid var(--line); display:grid; grid-template-columns:1fr 360px; gap:15px; align-items:center; background:#fafbfb; }
    .systems { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); }
    .system { min-width:0; padding:12px 14px 14px; }
    .system + .system { border-left:1px solid var(--line); }
    .system-title { display:flex; justify-content:space-between; gap:8px; margin-bottom:8px; font-size:15px; font-weight:700; }
    .model { color:var(--reveal); font-size:12px; }
    audio { display:block; width:100%; height:36px; }
    .ratings { padding:10px 14px; border-top:1px solid var(--line); display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:12px 20px; }
    .rating { min-width:0; } .rating strong { display:block; margin-bottom:5px; color:var(--accent); }
    .options { display:flex; gap:12px; flex-wrap:wrap; } .options label { cursor:pointer; }
    .case-actions { padding:0 14px 12px; display:flex; justify-content:flex-end; }
    @media(max-width:880px) { .toolbar,.head,.reference,.systems,.ratings { grid-template-columns:1fr; } .system + .system { border-left:0; border-top:1px solid var(--line); } }
  </style>
</head>
<body>
<header><div class="toolbar"><div><h1>Original VD / SFT VD / Base ICL</h1><div class="summary" id="summary"></div></div><button id="reveal-all" type="button">Reveal all</button><button id="export" type="button">Export ratings</button></div></header>
<main id="root"></main>
<script>
const ROWS=__ROWS__;
const STORE="vd-sft-icl-three-way-v1";
let ratings=JSON.parse(localStorage.getItem(STORE)||"{}");
const esc=value=>String(value).replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
function setRating(id,metric,value){ratings[id]??={};ratings[id][metric]=value;localStorage.setItem(STORE,JSON.stringify(ratings));summary();}
function summary(){const total=ROWS.length*3;const count=Object.values(ratings).reduce((n,row)=>n+Object.values(row).filter(Boolean).length,0);document.getElementById("summary").textContent=`${ROWS.length} cases · ${count}/${total} ratings · listen to reference first`;}
function metric(row,index,name,title){const choices=[...row.systems.map(s=>s.system_id),"tie"];return `<div class="rating"><strong>${title}</strong><div class="options">${choices.map(value=>`<label><input type="radio" name="${name}-${index}" data-metric="${name}" value="${value}" ${ratings[row.sample_id]?.[name]===value?"checked":""}> ${value==="tie"?"Tie":"System "+value}</label>`).join("")}</div></div>`;}
function render(){document.getElementById("root").innerHTML=ROWS.map((row,index)=>{
  const systems=row.systems.map(system=>`<section class="system"><div class="system-title"><span>System ${system.system_id}</span><span class="model" hidden>${esc(system.model)}</span></div><audio controls preload="metadata" src="${esc(system.audio_src)}"></audio></section>`).join("");
  return `<article data-id="${esc(row.sample_id)}"><div class="head"><div><span class="label">Case / emotion</span><div class="case">${esc(row.sample_id)}</div><div>${esc(row.emotion)}</div></div><div><span class="label">Target text</span>${esc(row.text)}</div><div><span class="label">Caption</span>${esc(row.instruction)}</div></div><div class="reference"><div><span class="label">Reference transcript</span>${esc(row.reference_text)}</div><audio controls preload="metadata" src="${esc(row.reference_src)}"></audio></div><div class="systems">${systems}</div><div class="ratings">${metric(row,index,"style","Caption / 表现力匹配")}${metric(row,index,"quality","自然度与音质")}${metric(row,index,"speaker","与 reference 音色相似")}</div><div class="case-actions"><button class="reveal" type="button">Reveal</button></div></article>`;
}).join("");
document.querySelectorAll('.ratings input').forEach(input=>input.addEventListener('change',event=>setRating(event.target.closest('article').dataset.id,event.target.dataset.metric,event.target.value)));
document.querySelectorAll('.reveal').forEach(button=>button.addEventListener('click',event=>{const labels=event.target.closest('article').querySelectorAll('.model');const show=[...labels].some(label=>label.hidden);labels.forEach(label=>label.hidden=!show);event.target.textContent=show?'Hide':'Reveal';}));summary();}
document.getElementById('reveal-all').addEventListener('click',()=>document.querySelectorAll('.model').forEach(label=>label.hidden=false));
document.getElementById('export').addEventListener('click',()=>{const blob=new Blob([JSON.stringify({ratings,rows:ROWS},null,2)],{type:'application/json'});const link=document.createElement('a');link.href=URL.createObjectURL(blob);link.download='vd_sft_icl_ratings.json';link.click();URL.revokeObjectURL(link.href);});
render();
</script>
</body>
</html>
"""


def write_report(output_dir: Path, cases: list[dict[str, Any]], seed: int) -> None:
    rows, key = build_blind_rows(cases, output_dir, seed)
    payload = json.dumps(rows, ensure_ascii=False).replace("<", "\\u003c")
    (output_dir / "listen.html").write_text(
        HTML_TEMPLATE.replace("__ROWS__", payload), encoding="utf-8"
    )
    (output_dir / "blind_key.json").write_text(
        json.dumps(key, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate SFT VoiceDesign audio and build a three-way blind comparison."
    )
    parser.add_argument("--baseline-manifest", type=Path, required=True)
    parser.add_argument("--sft-model-path", type=Path, required=True)
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

    cases = read_jsonl(args.baseline_manifest.expanduser().resolve())
    if args.limit < 0 or args.limit > len(cases):
        parser.error(f"--limit must be between 0 and {len(cases)}")
    if args.limit:
        cases = cases[: args.limit]
    validate_baseline_cases(cases)
    output_dir = args.output_dir.expanduser().resolve()
    audio_dir = output_dir / "audio" / "sft_vd"
    audio_dir.mkdir(parents=True, exist_ok=True)

    import numpy as np
    import soundfile as sf
    import torch
    from qwen_tts import Qwen3TTSModel

    sft = Qwen3TTSModel.from_pretrained(
        str(args.sft_model_path.expanduser().resolve()),
        device_map=args.device,
        dtype=getattr(torch, args.dtype),
        attn_implementation=args.attn_implementation,
    )
    if sft.model.tts_model_type != "voice_design":
        raise ValueError("SFT model must be a VoiceDesign checkpoint")
    kwargs = generation_kwargs(args)
    for index, case in enumerate(cases, 1):
        started = time.monotonic()
        seed = sample_seed(args.seed, case["sample_id"])
        set_seed(seed, torch, np)
        wavs, sample_rate = sft.generate_voice_design(
            text=case["text"],
            instruct=case["instruction"],
            language=case.get("language", "Chinese"),
            non_streaming_mode=True,
            **kwargs,
        )
        output = audio_dir / f"{index:02d}_{case['sample_id']}.wav"
        sf.write(output, wavs[0], sample_rate)
        case["sft_vd_audio"] = str(output.resolve())
        case["sft_seed"] = seed
        case["sft_conditioning"] = "target_text + caption"
        print(
            json.dumps(
                {
                    "system": "sft_vd",
                    "case": index,
                    "name": case["sample_id"],
                    "seconds": round(time.monotonic() - started, 2),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    write_manifest(output_dir / "manifest.jsonl", cases)
    write_report(output_dir, cases, args.seed)
    print(
        json.dumps(
            {
                "cases": len(cases),
                "listen_html": str(output_dir / "listen.html"),
                "blind_key": str(output_dir / "blind_key.json"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
