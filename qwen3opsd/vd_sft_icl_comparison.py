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
from qwen3opsd.gap_eval import build_gap_cases


def build_html(path: Path, cases: list[dict[str, Any]]) -> None:
    rows = [
        {
            "sample_id": case["sample_id"],
            "emotion": case.get("emotion", ""),
            "text": case["text"],
            "instruction": case["instruction"],
            "reference_text": case["teacher_ref_text"],
            "reference_src": relative_path(case["teacher_ref_audio"], path.parent),
            "original_vd_src": relative_path(case["original_vd_audio"], path.parent),
            "sft_vd_src": relative_path(case["sft_vd_audio"], path.parent),
            "base_icl_caption_src": relative_path(
                case["base_icl_caption_audio"], path.parent
            ),
        }
        for case in cases
    ]
    payload = json.dumps(rows, ensure_ascii=False).replace("<", "\\u003c")
    html = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Original VD / SFT VD / Base ICL + Caption</title>
  <style>
    :root { color-scheme:light; --ink:#17191c; --muted:#626870; --line:#d7dce1; --bg:#f3f4f5; --surface:#fff; --original:#8b4a20; --sft:#176b45; --icl:#315f91; }
    * { box-sizing:border-box; }
    body { margin:0; background:var(--bg); color:var(--ink); font:14px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif; letter-spacing:0; }
    header { position:sticky; top:0; z-index:2; border-bottom:1px solid var(--line); background:rgba(255,255,255,.97); }
    .bar,main { width:min(1320px,calc(100% - 28px)); margin:auto; }
    .bar { padding:12px 0; }
    h1 { margin:0; font-size:18px; }
    .sub,.label { color:var(--muted); font-size:12px; }
    main { padding:14px 0 50px; }
    article { margin-bottom:12px; border:1px solid var(--line); border-radius:6px; background:var(--surface); overflow:hidden; }
    .meta { display:grid; grid-template-columns:150px minmax(220px,.9fr) minmax(320px,1.25fr); gap:14px; padding:11px 13px; border-bottom:1px solid var(--line); }
    .label { display:block; margin-bottom:3px; font-weight:700; }
    .case { overflow-wrap:anywhere; font:12px ui-monospace,SFMono-Regular,Consolas,monospace; }
    .reference { display:grid; grid-template-columns:1fr 360px; gap:14px; align-items:center; padding:10px 13px; border-bottom:1px solid var(--line); background:#fafbfb; }
    .systems { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); }
    .system { min-width:0; padding:12px 13px 14px; }
    .system + .system { border-left:1px solid var(--line); }
    .title { margin-bottom:2px; font-size:15px; font-weight:700; }
    .conditioning { min-height:36px; margin-bottom:7px; color:var(--muted); font-size:12px; }
    .original { color:var(--original); } .sft { color:var(--sft); } .icl { color:var(--icl); }
    audio { display:block; width:100%; height:36px; }
    @media(max-width:860px) { .meta,.reference,.systems { grid-template-columns:1fr; } .system + .system { border-left:0; border-top:1px solid var(--line); } .conditioning { min-height:0; } }
  </style>
</head>
<body>
<header><div class="bar"><h1>Original VD / SFT VD / Base ICL + Caption</h1><div class="sub">相同 target、caption 与逐 case 随机种子；Base ICL 使用 non_streaming=True 和联合 codec 解码</div></div></header>
<main id="root"></main>
<script>
const rows=__ROWS__;
const esc=v=>String(v).replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
document.getElementById("root").innerHTML=rows.map(row=>`<article>
  <div class="meta"><div><span class="label">Case / emotion</span><div class="case">${esc(row.sample_id)}</div><div>${esc(row.emotion)}</div></div><div><span class="label">Target text</span>${esc(row.text)}</div><div><span class="label">Caption</span>${esc(row.instruction)}</div></div>
  <div class="reference"><div><span class="label">Reference transcript</span>${esc(row.reference_text)}</div><audio controls preload="metadata" src="${esc(row.reference_src)}"></audio></div>
  <div class="systems">
    <section class="system"><div class="title original">Original VoiceDesign</div><div class="conditioning">target text + caption</div><audio controls preload="metadata" src="${esc(row.original_vd_src)}"></audio></section>
    <section class="system"><div class="title sft">SFT VoiceDesign</div><div class="conditioning">target text + caption</div><audio controls preload="metadata" src="${esc(row.sft_vd_src)}"></audio></section>
    <section class="system"><div class="title icl">Base ICL + Caption</div><div class="conditioning">target text + caption + reference audio + transcript</div><audio controls preload="metadata" src="${esc(row.base_icl_caption_src)}"></audio></section>
  </div>
</article>`).join("");
</script>
</body>
</html>""".replace("__ROWS__", payload)
    path.write_text(html, encoding="utf-8")


def load_model(path: Path, *, device: str, dtype, attn_implementation: str):
    from qwen_tts import Qwen3TTSModel

    return Qwen3TTSModel.from_pretrained(
        str(path.expanduser().resolve()),
        device_map=device,
        dtype=dtype,
        attn_implementation=attn_implementation,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a corrected named three-way comparison.")
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--original-vd-path", type=Path, required=True)
    parser.add_argument("--sft-vd-path", type=Path, required=True)
    parser.add_argument("--base-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    rows = read_jsonl(args.input_jsonl.expanduser().resolve())
    cases = build_gap_cases(rows, args.limit)
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    audio_dirs = {
        "original_vd": output_dir / "audio" / "original_vd",
        "sft_vd": output_dir / "audio" / "sft_vd",
        "base_icl_caption": output_dir / "audio" / "base_icl_caption",
    }
    for directory in audio_dirs.values():
        directory.mkdir(parents=True, exist_ok=True)

    import numpy as np
    import soundfile as sf
    import torch

    dtype = getattr(torch, args.dtype)
    kwargs = generation_kwargs(args)
    vd_systems = (
        ("original_vd", args.original_vd_path),
        ("sft_vd", args.sft_vd_path),
    )
    for system, model_path in vd_systems:
        model = load_model(
            model_path,
            device=args.device,
            dtype=dtype,
            attn_implementation=args.attn_implementation,
        )
        if model.model.tts_model_type != "voice_design":
            raise ValueError(f"{system} checkpoint must be VoiceDesign")
        for index, case in enumerate(cases, 1):
            seed = sample_seed(args.seed, case["sample_id"])
            set_seed(seed, torch, np)
            started = time.monotonic()
            wavs, sample_rate = model.generate_voice_design(
                text=case["text"],
                instruct=case["instruction"],
                language=case.get("language", "Chinese"),
                non_streaming_mode=True,
                **kwargs,
            )
            output = audio_dirs[system] / f"{index:02d}_{case['sample_id']}.wav"
            sf.write(output, wavs[0], sample_rate)
            case[f"{system}_audio"] = str(output.resolve())
            print(
                json.dumps(
                    {"system": system, "case": index, "seconds": round(time.monotonic() - started, 2)},
                    ensure_ascii=False,
                ),
                flush=True,
            )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    base = load_model(
        args.base_path,
        device=args.device,
        dtype=dtype,
        attn_implementation=args.attn_implementation,
    )
    if base.model.tts_model_type != "base":
        raise ValueError("Base ICL checkpoint must be a Base model")
    for index, case in enumerate(cases, 1):
        prompt_items = base.create_voice_clone_prompt(
            ref_audio=case["teacher_ref_audio"],
            ref_text=case["teacher_ref_text"],
            x_vector_only_mode=False,
        )
        prompt = base._prompt_items_to_voice_clone_prompt(prompt_items)
        seed = sample_seed(args.seed, case["sample_id"])
        set_seed(seed, torch, np)
        started = time.monotonic()
        wavs, sample_rate = generate_icl_teacher(
            base,
            text=case["text"],
            instruction=case["instruction"],
            reference_text=case["teacher_ref_text"],
            language=case.get("language", "Chinese"),
            voice_clone_prompt=prompt,
            non_streaming_mode=True,
            **kwargs,
        )
        output = audio_dirs["base_icl_caption"] / f"{index:02d}_{case['sample_id']}.wav"
        sf.write(output, wavs[0], sample_rate)
        case["base_icl_caption_audio"] = str(output.resolve())
        print(
            json.dumps(
                {"system": "base_icl_caption", "case": index, "seconds": round(time.monotonic() - started, 2)},
                ensure_ascii=False,
            ),
            flush=True,
        )

    write_manifest(output_dir / "manifest.jsonl", cases)
    build_html(output_dir / "listen.html", cases)
    print(json.dumps({"listen_html": str(output_dir / "listen.html")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
