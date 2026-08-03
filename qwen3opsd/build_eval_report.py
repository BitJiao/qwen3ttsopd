from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any

from qwen3opsd.eval_emotiontalk import latest_manifest_rows, read_jsonl


def parse_run_spec(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("run must use LABEL=/path/to/manifest.jsonl")
    label, raw_path = value.split("=", 1)
    label = label.strip()
    if not label or not raw_path.strip():
        raise argparse.ArgumentTypeError("run label and manifest path must be non-empty")
    return label, Path(raw_path).expanduser().resolve()


def successful_rows(path: Path) -> dict[str, dict[str, Any]]:
    return {
        sample_id: row
        for sample_id, row in latest_manifest_rows(path).items()
        if row.get("status") == "ok"
        and row.get("generated_audio")
        and Path(row["generated_audio"]).is_file()
    }


def relative_audio_path(audio_path: str, report_dir: Path) -> str:
    return Path(os.path.relpath(Path(audio_path).resolve(), report_dir.resolve())).as_posix()


def build_report_rows(
    source_rows: list[dict[str, Any]],
    runs: list[tuple[str, Path]],
    report_dir: Path,
    seed: int,
    limit: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, int]]:
    loaded = [(label, successful_rows(path)) for label, path in runs]
    counts = {label: len(rows) for label, rows in loaded}
    complete_ids = set.intersection(*(set(rows) for _, rows in loaded))

    report_rows: list[dict[str, Any]] = []
    blind_key: dict[str, Any] = {}
    for source in source_rows:
        sample_id = str(source["sample_id"])
        if sample_id not in complete_ids:
            continue
        candidates = []
        for label, rows in loaded:
            row = rows[sample_id]
            candidates.append(
                {
                    "run_label": label,
                    "model_name": str(row.get("model_name", label)),
                    "conditioning": str(row.get("conditioning", "")),
                    "audio_src": relative_audio_path(row["generated_audio"], report_dir),
                    "audio_path": str(Path(row["generated_audio"]).resolve()),
                    "audio_seconds": row.get("audio_seconds"),
                }
            )
        sample_rng = random.Random(f"{seed}:{sample_id}")
        sample_rng.shuffle(candidates)
        for index, candidate in enumerate(candidates):
            candidate["system_id"] = chr(ord("A") + index)

        blind_key[sample_id] = {
            candidate["system_id"]: {
                "run_label": candidate["run_label"],
                "model_name": candidate["model_name"],
                "conditioning": candidate["conditioning"],
            }
            for candidate in candidates
        }
        report_rows.append(
            {
                "sample_id": sample_id,
                "source_id": str(source.get("source_id", sample_id)),
                "text": str(source.get("text", "")),
                "instruction": str(source.get("instruction", "")),
                "emotion": str(source.get("emotion", "")),
                "task": str(source.get("task", "")),
                "group": str(source.get("task") or source.get("emotion", "")),
                "speaker_id": str(source.get("speaker_id", "")),
                "target_audio": str(Path(source["target_audio"]).resolve()),
                "student_spk_audio": str(Path(source["student_spk_audio"]).resolve()),
                "target_src": relative_audio_path(source["target_audio"], report_dir),
                "enrollment_src": relative_audio_path(source["student_spk_audio"], report_dir),
                "systems": candidates,
            }
        )
        if limit > 0 and len(report_rows) >= limit:
            break
    return report_rows, blind_key, counts


def write_api_jsonl(path: Path, report_rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in report_rows:
            output = {
                "sample_id": row["sample_id"],
                "source_id": row["source_id"],
                "text": row["text"],
                "instruction": row["instruction"],
                "emotion": row["emotion"],
                "task": row["task"],
                "speaker_id": row["speaker_id"],
                "target_audio": row["target_audio"],
                "student_spk_audio": row["student_spk_audio"],
                "candidates": [
                    {
                        "run_label": system["run_label"],
                        "model_name": system["model_name"],
                        "conditioning": system["conditioning"],
                        "generated_audio": system["audio_path"],
                    }
                    for system in row["systems"]
                ],
            }
            handle.write(json.dumps(output, ensure_ascii=False) + "\n")


def write_gemini_jsonls(output_dir: Path, report_rows: list[dict[str, Any]]) -> dict[str, str]:
    by_run: dict[str, dict[str, dict[str, Any]]] = {}
    for row in report_rows:
        task = str(row.get("task", ""))
        if task not in {"APS", "DSD", "RP"}:
            continue
        for system in row["systems"]:
            run_label = str(system["run_label"])
            records = by_run.setdefault(run_label, {})
            source_id = str(row["source_id"])
            record = records.setdefault(source_id, {"id": source_id, "text": row["text"]})
            record[task] = {
                "instruction": row["instruction"],
                "gen_path": system["audio_path"],
            }

    outputs: dict[str, str] = {}
    for run_label, records in by_run.items():
        filename = "gemini_" + "".join(
            character if character.isalnum() or character in "._-" else "_"
            for character in run_label
        ) + ".jsonl"
        path = output_dir / filename
        with path.open("w", encoding="utf-8") as handle:
            for record in records.values():
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        outputs[run_label] = str(path)
    return outputs


HTML_TEMPLATE = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>EmotionTalk Blind Listening Evaluation</title>
  <style>
    :root { color-scheme: light; --ink: #17191c; --muted: #626870; --line: #d9dde2; --bg: #f4f5f6; --surface: #fff; --accent: #176b45; --warn: #8c4f13; }
    * { box-sizing: border-box; }
    body { margin: 0; background: var(--bg); color: var(--ink); font: 14px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif; letter-spacing: 0; }
    header { position: sticky; top: 0; z-index: 10; border-bottom: 1px solid var(--line); background: rgba(255,255,255,.96); }
    .toolbar { max-width: 1440px; margin: 0 auto; padding: 12px 20px; display: grid; grid-template-columns: minmax(220px, 1fr) auto auto auto; gap: 10px; align-items: center; }
    h1 { margin: 0; font-size: 18px; font-weight: 650; }
    .summary { color: var(--muted); font-size: 12px; }
    input, select, button { min-height: 34px; border: 1px solid #bcc2c9; border-radius: 4px; background: #fff; color: var(--ink); padding: 6px 10px; font: inherit; letter-spacing: 0; }
    button { cursor: pointer; font-weight: 600; }
    button:hover { border-color: #6b737c; }
    main { max-width: 1440px; margin: 0 auto; padding: 16px 20px 64px; }
    article { margin: 0 0 14px; border: 1px solid var(--line); border-radius: 6px; background: var(--surface); overflow: hidden; }
    .sample-head { padding: 12px 14px; border-bottom: 1px solid var(--line); display: grid; grid-template-columns: 90px minmax(160px, .7fr) minmax(280px, 1.4fr); gap: 14px; }
    .sample-id { color: var(--muted); font: 12px ui-monospace, SFMono-Regular, Consolas, monospace; overflow-wrap: anywhere; }
    .label { display: block; margin-bottom: 3px; color: var(--muted); font-size: 11px; font-weight: 700; text-transform: uppercase; }
    .refs { padding: 9px 14px; border-bottom: 1px solid var(--line); display: flex; gap: 18px; align-items: center; flex-wrap: wrap; }
    .refs audio { width: 250px; height: 34px; }
    .systems { display: grid; grid-template-columns: repeat(var(--system-count), minmax(240px, 1fr)); }
    .system { min-width: 0; padding: 12px 14px 14px; border-right: 1px solid var(--line); }
    .system:last-child { border-right: 0; }
    .system-title { display: flex; justify-content: space-between; gap: 8px; align-items: baseline; margin-bottom: 8px; }
    .system-name { font-size: 15px; font-weight: 700; }
    .model-name { color: var(--warn); font-size: 11px; font-weight: 650; }
    audio { display: block; width: 100%; height: 36px; }
    .rating { margin-top: 10px; display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
    .rating label { color: var(--muted); font-size: 12px; }
    .rating select { display: block; width: 100%; margin-top: 3px; }
    .preference { padding: 9px 14px; border-top: 1px solid var(--line); display: flex; align-items: center; gap: 14px; flex-wrap: wrap; }
    .preference label { cursor: pointer; }
    .empty { padding: 48px 20px; text-align: center; color: var(--muted); }
    @media (max-width: 900px) {
      .toolbar { grid-template-columns: 1fr 1fr; }
      h1 { grid-column: 1 / -1; }
      .sample-head { grid-template-columns: 1fr; }
      .systems { grid-template-columns: 1fr; }
      .system { border-right: 0; border-bottom: 1px solid var(--line); }
      .system:last-child { border-bottom: 0; }
    }
  </style>
</head>
<body>
<header>
  <div class="toolbar">
    <div><h1>EmotionTalk Blind Listening</h1><div class="summary" id="summary"></div></div>
    <input id="search" type="search" placeholder="Search text, instruction, ID">
    <select id="emotion"><option value="">All groups</option></select>
    <button id="export" type="button">Export ratings</button>
  </div>
</header>
<main id="samples"></main>
<script>
const REPORT = __REPORT_DATA__;
const STORAGE_KEY = "emotiontalk-eval-ratings-v1";
let ratings = JSON.parse(localStorage.getItem(STORAGE_KEY) || "{}");
const root = document.getElementById("samples");
const search = document.getElementById("search");
const emotion = document.getElementById("emotion");

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
}
function scoreOptions(selected) {
  return '<option value="">-</option>' + [1,2,3,4,5].map(v => `<option value="${v}" ${String(selected) === String(v) ? "selected" : ""}>${v}</option>`).join("");
}
function saveRating(sampleId, systemId, field, value) {
  ratings[sampleId] ||= { systems: {}, preferred: "" };
  ratings[sampleId].systems ||= {};
  ratings[sampleId].systems[systemId] ||= {};
  ratings[sampleId].systems[systemId][field] = value ? Number(value) : null;
  localStorage.setItem(STORAGE_KEY, JSON.stringify(ratings));
  updateSummary();
}
function savePreference(sampleId, value) {
  ratings[sampleId] ||= { systems: {}, preferred: "" };
  ratings[sampleId].preferred = value;
  localStorage.setItem(STORAGE_KEY, JSON.stringify(ratings));
  updateSummary();
}
function render() {
  const query = search.value.trim().toLowerCase();
  const selectedEmotion = emotion.value;
  const visible = REPORT.filter(row => {
    const haystack = `${row.sample_id} ${row.text} ${row.instruction}`.toLowerCase();
    return (!query || haystack.includes(query)) && (!selectedEmotion || row.group === selectedEmotion);
  });
  if (!visible.length) {
    root.innerHTML = '<div class="empty">No matching samples.</div>';
    updateSummary(0);
    return;
  }
  root.innerHTML = visible.map((row, index) => {
    const saved = ratings[row.sample_id] || { systems: {}, preferred: "" };
    const systems = row.systems.map(system => {
      const score = (saved.systems || {})[system.system_id] || {};
      return `<section class="system">
        <div class="system-title"><span class="system-name">System ${system.system_id}</span><span class="model-name" hidden>${escapeHtml(system.run_label)}</span></div>
        <audio controls preload="none" src="${escapeHtml(system.audio_src)}"></audio>
        <div class="rating">
          <label>Instruction match<select data-sample="${escapeHtml(row.sample_id)}" data-system="${system.system_id}" data-field="instruction">${scoreOptions(score.instruction)}</select></label>
          <label>Audio quality<select data-sample="${escapeHtml(row.sample_id)}" data-system="${system.system_id}" data-field="quality">${scoreOptions(score.quality)}</select></label>
        </div>
      </section>`;
    }).join("");
    const choices = row.systems.map(system => `<label><input type="radio" name="pref-${index}" value="${system.system_id}" ${saved.preferred === system.system_id ? "checked" : ""}> System ${system.system_id}</label>`).join("");
    return `<article data-group="${escapeHtml(row.group)}" style="--system-count:${row.systems.length}">
      <div class="sample-head">
        <div><span class="label">Sample</span><div class="sample-id">${escapeHtml(row.sample_id)}</div><div>${escapeHtml(row.group)} / spk ${escapeHtml(row.speaker_id)}</div></div>
        <div><span class="label">Text</span>${escapeHtml(row.text)}</div>
        <div><span class="label">Instruction</span>${escapeHtml(row.instruction)}</div>
      </div>
      <div class="refs"><span><span class="label">Enrollment</span><audio controls preload="none" src="${escapeHtml(row.enrollment_src)}"></audio></span><span><span class="label">Ground truth</span><audio controls preload="none" src="${escapeHtml(row.target_src)}"></audio></span></div>
      <div class="systems">${systems}</div>
      <div class="preference"><strong>Preferred:</strong>${choices}<label><input type="radio" name="pref-${index}" value="tie" ${saved.preferred === "tie" ? "checked" : ""}> Tie</label><button type="button" class="reveal">Reveal systems</button></div>
    </article>`;
  }).join("");
  root.querySelectorAll("select[data-field]").forEach(select => select.addEventListener("change", event => {
    const el = event.currentTarget;
    saveRating(el.dataset.sample, el.dataset.system, el.dataset.field, el.value);
  }));
  root.querySelectorAll('.preference input[type="radio"]').forEach(input => input.addEventListener("change", event => {
    const article = event.currentTarget.closest("article");
    const sampleId = article.querySelector("select[data-sample]").dataset.sample;
    savePreference(sampleId, event.currentTarget.value);
  }));
  root.querySelectorAll(".reveal").forEach(button => button.addEventListener("click", event => {
    const labels = event.currentTarget.closest("article").querySelectorAll(".model-name");
    const reveal = Array.from(labels).some(label => label.hidden);
    labels.forEach(label => label.hidden = !reveal);
    event.currentTarget.textContent = reveal ? "Hide systems" : "Reveal systems";
  }));
  updateSummary(visible.length);
}
function updateSummary(visibleCount) {
  const rated = Object.values(ratings).filter(value => value && value.preferred).length;
  const count = visibleCount === undefined ? document.querySelectorAll("article").length : visibleCount;
  document.getElementById("summary").textContent = `${count} visible / ${REPORT.length} complete samples; ${rated} preferences saved`;
}

[...new Set(REPORT.map(row => row.group).filter(Boolean))].sort().forEach(value => {
  const option = document.createElement("option"); option.value = value; option.textContent = value; emotion.appendChild(option);
});
search.addEventListener("input", render);
emotion.addEventListener("change", render);
document.getElementById("export").addEventListener("click", () => {
  const blob = new Blob([JSON.stringify({ exported_at: new Date().toISOString(), ratings }, null, 2)], {type: "application/json"});
  const link = document.createElement("a"); link.href = URL.createObjectURL(blob); link.download = "emotiontalk_ratings.json"; link.click(); URL.revokeObjectURL(link.href);
});
render();
</script>
</body>
</html>
"""


def write_html(path: Path, report_rows: list[dict[str, Any]]) -> None:
    payload = json.dumps(report_rows, ensure_ascii=False).replace("<", "\\u003c")
    path.write_text(HTML_TEMPLATE.replace("__REPORT_DATA__", payload), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a blind listening page from EmotionTalk manifests.")
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--run", action="append", type=parse_run_spec, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--limit", type=int, default=0, help="0 includes every completed sample.")
    args = parser.parse_args()

    if len(args.run) < 2:
        parser.error("at least two --run arguments are required")
    labels = [label for label, _ in args.run]
    if len(labels) != len(set(labels)):
        parser.error("run labels must be unique")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    source_rows = read_jsonl(args.input_jsonl.expanduser().resolve())
    report_rows, blind_key, counts = build_report_rows(
        source_rows, args.run, output_dir, args.seed, args.limit
    )
    if not report_rows:
        raise ValueError("the supplied manifests have no successful sample in common")

    write_html(output_dir / "listen.html", report_rows)
    write_api_jsonl(output_dir / "api_judge.jsonl", report_rows)
    gemini_inputs = write_gemini_jsonls(output_dir, report_rows)
    (output_dir / "blind_key.json").write_text(
        json.dumps(blind_key, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    summary = {
        "complete_samples": len(report_rows),
        "successful_rows_per_run": counts,
        "listen_html": str(output_dir / "listen.html"),
        "api_judge_jsonl": str(output_dir / "api_judge.jsonl"),
        "blind_key": str(output_dir / "blind_key.json"),
        "gemini_inputs": gemini_inputs,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
