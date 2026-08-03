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


def build_html(path: Path, cases: list[dict[str, Any]]) -> None:
    rows = []
    for case in cases:
        rows.append(
            {
                "sample_id": case["sample_id"],
                "emotion": case.get("emotion", ""),
                "text": case["text"],
                "instruction": case["instruction"],
                "reference_text": case["teacher_ref_text"],
                "reference_src": relative_path(case["teacher_ref_audio"], path.parent),
                "streaming_src": relative_path(case["icl_caption_streaming_audio"], path.parent),
                "non_streaming_src": relative_path(
                    case["icl_caption_non_streaming_audio"], path.parent
                ),
            }
        )
    payload = json.dumps(rows, ensure_ascii=False).replace("<", "\\u003c")
    html = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Base ICL + Caption 对齐模式消融</title>
  <style>
    :root { color-scheme:light; --ink:#17191c; --muted:#626870; --line:#d7dce1; --bg:#f3f4f5; --surface:#fff; --accent:#176b45; }
    * { box-sizing:border-box; }
    body { margin:0; background:var(--bg); color:var(--ink); font:14px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif; letter-spacing:0; }
    header { position:sticky; top:0; z-index:2; border-bottom:1px solid var(--line); background:rgba(255,255,255,.97); }
    .bar,main { width:min(1180px,calc(100% - 28px)); margin:auto; }
    .bar { padding:12px 0; }
    h1 { margin:0; font-size:18px; }
    .sub,.label { color:var(--muted); font-size:12px; }
    main { padding:14px 0 50px; }
    article { margin-bottom:12px; border:1px solid var(--line); border-radius:6px; background:var(--surface); overflow:hidden; }
    .meta { display:grid; grid-template-columns:150px 1fr 1.25fr; gap:14px; padding:11px 13px; border-bottom:1px solid var(--line); }
    .label { display:block; margin-bottom:3px; font-weight:700; }
    .case { overflow-wrap:anywhere; font:12px ui-monospace,SFMono-Regular,Consolas,monospace; }
    .reference { display:grid; grid-template-columns:1fr 340px; gap:14px; align-items:center; padding:10px 13px; border-bottom:1px solid var(--line); background:#fafbfb; }
    .systems { display:grid; grid-template-columns:1fr 1fr; }
    .system { min-width:0; padding:12px 13px 14px; }
    .system + .system { border-left:1px solid var(--line); }
    .title { margin-bottom:8px; font-size:15px; font-weight:700; }
    .recommended { color:var(--accent); }
    audio { display:block; width:100%; height:36px; }
    @media(max-width:760px) { .meta,.reference,.systems { grid-template-columns:1fr; } .system + .system { border-left:0; border-top:1px solid var(--line); } }
  </style>
</head>
<body>
<header><div class="bar"><h1>Base ICL + Caption 对齐模式消融</h1><div class="sub">相同 reference、caption、target text、随机种子；两路均使用联合 codec 解码</div></div></header>
<main id="root"></main>
<script>
const rows=__ROWS__;
const esc=v=>String(v).replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
document.getElementById("root").innerHTML=rows.map(row=>`<article>
  <div class="meta"><div><span class="label">Case / emotion</span><div class="case">${esc(row.sample_id)}</div><div>${esc(row.emotion)}</div></div><div><span class="label">Target text</span>${esc(row.text)}</div><div><span class="label">Caption</span>${esc(row.instruction)}</div></div>
  <div class="reference"><div><span class="label">Reference transcript</span>${esc(row.reference_text)}</div><audio controls preload="metadata" src="${esc(row.reference_src)}"></audio></div>
  <div class="systems"><section class="system"><div class="title recommended">False · 官方默认流式文本对齐</div><audio controls preload="metadata" src="${esc(row.streaming_src)}"></audio></section><section class="system"><div class="title">True · 非流式完整文本对齐</div><audio controls preload="metadata" src="${esc(row.non_streaming_src)}"></audio></section></div>
</article>`).join("");
</script>
</body>
</html>""".replace("__ROWS__", payload)
    path.write_text(html, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare ICL text-alignment modes.")
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    cases = read_jsonl(args.input_manifest.expanduser().resolve())
    if args.limit > 0:
        cases = cases[: args.limit]
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    audio_dirs = {
        False: output_dir / "audio" / "streaming_false",
        True: output_dir / "audio" / "non_streaming_true",
    }
    for directory in audio_dirs.values():
        directory.mkdir(parents=True, exist_ok=True)

    import numpy as np
    import soundfile as sf
    import torch
    from qwen_tts import Qwen3TTSModel

    model = Qwen3TTSModel.from_pretrained(
        str(args.model_path.expanduser().resolve()),
        device_map=args.device,
        dtype=getattr(torch, args.dtype),
        attn_implementation=args.attn_implementation,
    )
    if model.model.tts_model_type != "base":
        raise ValueError("ICL ablation requires a Base checkpoint")
    kwargs = generation_kwargs(args)
    for index, case in enumerate(cases, 1):
        prompt_items = model.create_voice_clone_prompt(
            ref_audio=case["teacher_ref_audio"],
            ref_text=case["teacher_ref_text"],
            x_vector_only_mode=False,
        )
        prompt = model._prompt_items_to_voice_clone_prompt(prompt_items)
        for non_streaming_mode in (False, True):
            seed = sample_seed(args.seed, case["sample_id"])
            set_seed(seed, torch, np)
            started = time.monotonic()
            wavs, sample_rate = generate_icl_teacher(
                model,
                text=case["text"],
                instruction=case["instruction"],
                reference_text=case["teacher_ref_text"],
                language=case.get("language", "Chinese"),
                voice_clone_prompt=prompt,
                non_streaming_mode=non_streaming_mode,
                **kwargs,
            )
            output = audio_dirs[non_streaming_mode] / f"{index:02d}_{case['sample_id']}.wav"
            sf.write(output, wavs[0], sample_rate)
            key = (
                "icl_caption_non_streaming_audio"
                if non_streaming_mode
                else "icl_caption_streaming_audio"
            )
            case[key] = str(output.resolve())
            print(
                json.dumps(
                    {
                        "case": index,
                        "mode": non_streaming_mode,
                        "seconds": round(time.monotonic() - started, 2),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    write_manifest(output_dir / "manifest.jsonl", cases)
    build_html(output_dir / "listen.html", cases)
    print(json.dumps({"listen_html": str(output_dir / "listen.html")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
