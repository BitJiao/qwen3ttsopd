from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from qwen3opsd.compare_vd_teacher import read_jsonl, relative_path


def audio_stats(path: str | Path) -> dict[str, float]:
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    rms = float(np.sqrt(np.mean(np.square(audio), dtype=np.float64)))
    peak = float(np.max(np.abs(audio)))
    return {
        "duration": audio.shape[0] / sample_rate,
        "mean_dbfs": 20.0 * math.log10(max(rms, 1e-12)),
        "peak_dbfs": 20.0 * math.log10(max(peak, 1e-12)),
    }


def select_by_length(cases: list[dict[str, Any]]) -> list[tuple[str, dict[str, Any]]]:
    ordered = sorted(cases, key=lambda case: case["reference_stats"]["duration"])
    middle_start = max(0, len(ordered) // 2 - 2)
    selected = [
        *(('Short reference', case) for case in ordered[:3]),
        *(('Medium reference', case) for case in ordered[middle_start : middle_start + 3]),
        *(('Long reference', case) for case in ordered[-3:]),
    ]
    seen: set[str] = set()
    return [
        (group, case)
        for group, case in selected
        if not (case["sample_id"] in seen or seen.add(case["sample_id"]))
    ]


def gain_compensate(source: Path, output: Path, gain_db: float) -> None:
    audio, sample_rate = sf.read(source, dtype="float32", always_2d=False)
    adjusted = audio * (10.0 ** (gain_db / 20.0))
    sf.write(output, adjusted, sample_rate)


def build_html(path: Path, rows: list[dict[str, Any]]) -> None:
    payload = json.dumps(rows, ensure_ascii=False).replace("<", "\\u003c")
    html = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Reference Length and ICL Loudness</title>
  <style>
    :root { color-scheme:light; --ink:#17191c; --muted:#626870; --line:#d7dce1; --bg:#f3f4f5; --surface:#fff; --accent:#176b45; --warn:#9b4b20; }
    * { box-sizing:border-box; }
    body { margin:0; background:var(--bg); color:var(--ink); font:14px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif; letter-spacing:0; }
    header { position:sticky; top:0; z-index:2; border-bottom:1px solid var(--line); background:rgba(255,255,255,.97); }
    .bar,main { width:min(1420px,calc(100% - 28px)); margin:auto; }
    .bar { padding:12px 0; } h1 { margin:0; font-size:18px; }
    .sub,.label,.stats { color:var(--muted); font-size:12px; }
    main { padding:14px 0 50px; }
    .group { margin:20px 0 8px; font-size:16px; }
    article { margin-bottom:12px; border:1px solid var(--line); border-radius:6px; background:var(--surface); overflow:hidden; }
    .meta { display:grid; grid-template-columns:150px minmax(220px,.9fr) minmax(320px,1.25fr); gap:14px; padding:11px 13px; border-bottom:1px solid var(--line); }
    .label { display:block; margin-bottom:3px; font-weight:700; }
    .case { overflow-wrap:anywhere; font:12px ui-monospace,SFMono-Regular,Consolas,monospace; }
    .reference { display:grid; grid-template-columns:1fr 420px; gap:14px; align-items:center; padding:10px 13px; border-bottom:1px solid var(--line); background:#fafbfb; }
    .systems { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); }
    .system { min-width:0; padding:12px 13px 14px; }
    .system + .system { border-left:1px solid var(--line); }
    .title { margin-bottom:2px; font-size:14px; font-weight:700; }
    .matched { color:var(--accent); } .raw { color:var(--warn); }
    .conditioning { min-height:38px; margin-bottom:7px; color:var(--muted); font-size:12px; }
    audio { display:block; width:100%; height:36px; }
    @media(max-width:980px) { .meta,.reference,.systems { grid-template-columns:1fr; } .system + .system { border-left:0; border-top:1px solid var(--line); } .conditioning { min-height:0; } }
  </style>
</head>
<body>
<header><div class="bar"><h1>Reference 时长与 ICL 响度对比</h1><div class="sub">短/中/长各 3 条；Gain-matched 仅调整输出幅度，不重新推理</div></div></header>
<main id="root"></main>
<script>
const rows=__ROWS__;
const esc=v=>String(v).replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const f=v=>Number(v).toFixed(1);
let last="";
document.getElementById("root").innerHTML=rows.map(row=>{
  const heading=row.group!==last?`<h2 class="group">${esc(row.group)}</h2>`:""; last=row.group;
  return `${heading}<article>
    <div class="meta"><div><span class="label">Case / emotion</span><div class="case">${esc(row.sample_id)}</div><div>${esc(row.emotion)}</div></div><div><span class="label">Target text</span>${esc(row.text)}</div><div><span class="label">Caption</span>${esc(row.instruction)}</div></div>
    <div class="reference"><div><span class="label">Reference · ${f(row.reference.duration)} s</span>${esc(row.reference_text)}<div class="stats">mean ${f(row.reference.mean_dbfs)} dBFS · peak ${f(row.reference.peak_dbfs)} dBFS</div></div><audio controls preload="metadata" src="${esc(row.reference.src)}"></audio></div>
    <div class="systems">
      <section class="system"><div class="title">Original VD</div><div class="conditioning">mean ${f(row.original.mean_dbfs)} dBFS</div><audio controls preload="metadata" src="${esc(row.original.src)}"></audio></section>
      <section class="system"><div class="title">SFT VD</div><div class="conditioning">mean ${f(row.sft.mean_dbfs)} dBFS</div><audio controls preload="metadata" src="${esc(row.sft.src)}"></audio></section>
      <section class="system"><div class="title raw">Base ICL + Caption · Raw</div><div class="conditioning">mean ${f(row.icl.mean_dbfs)} dBFS · 比 reference ${f(row.icl.mean_dbfs-row.reference.mean_dbfs)} dB</div><audio controls preload="metadata" src="${esc(row.icl.src)}"></audio></section>
      <section class="system"><div class="title matched">Base ICL · Gain compensated</div><div class="conditioning">+${f(row.gain_db)} dB；峰值限制在 -1 dBFS 以下</div><audio controls preload="metadata" src="${esc(row.matched_src)}"></audio></section>
    </div>
  </article>`;
}).join("");
</script>
</body>
</html>""".replace("__ROWS__", payload)
    path.write_text(html, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a length/loudness listening report.")
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    cases = read_jsonl(args.input_manifest.expanduser().resolve())
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    matched_dir = output_dir / "audio" / "icl_gain_compensated"
    matched_dir.mkdir(parents=True, exist_ok=True)

    for case in cases:
        case["reference_stats"] = audio_stats(case["teacher_ref_audio"])
    selected = select_by_length(cases)
    rows: list[dict[str, Any]] = []
    for index, (group, case) in enumerate(selected, 1):
        reference = {**case["reference_stats"]}
        original = audio_stats(case["original_vd_audio"])
        sft = audio_stats(case["sft_vd_audio"])
        icl = audio_stats(case["base_icl_caption_audio"])
        desired_gain = reference["mean_dbfs"] - icl["mean_dbfs"]
        peak_safe_gain = -1.0 - icl["peak_dbfs"]
        gain_db = max(0.0, min(desired_gain, peak_safe_gain))
        matched = matched_dir / f"{index:02d}_{case['sample_id']}.wav"
        gain_compensate(Path(case["base_icl_caption_audio"]), matched, gain_db)
        reference["src"] = relative_path(case["teacher_ref_audio"], output_dir)
        original["src"] = relative_path(case["original_vd_audio"], output_dir)
        sft["src"] = relative_path(case["sft_vd_audio"], output_dir)
        icl["src"] = relative_path(case["base_icl_caption_audio"], output_dir)
        rows.append(
            {
                "group": group,
                "sample_id": case["sample_id"],
                "emotion": case.get("emotion", ""),
                "text": case["text"],
                "instruction": case["instruction"],
                "reference_text": case["teacher_ref_text"],
                "reference": reference,
                "original": original,
                "sft": sft,
                "icl": icl,
                "gain_db": gain_db,
                "matched_src": relative_path(matched, output_dir),
            }
        )

    (output_dir / "report.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    build_html(output_dir / "listen.html", rows)
    print(json.dumps({"cases": len(rows), "listen_html": str(output_dir / "listen.html")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
