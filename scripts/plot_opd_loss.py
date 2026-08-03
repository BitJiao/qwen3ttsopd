#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot OPD loss curves from the JSONL-style training log.")
    parser.add_argument("--log", type=Path, default=Path("logs/emotiontalk_opd.log"))
    parser.add_argument("--output", type=Path, default=Path("outputs/emotiontalk_opd_loss.png"))
    parser.add_argument("--window", type=int, default=50, help="EMA span in training steps.")
    return parser.parse_args()


def load_records(path: Path) -> list[dict]:
    by_step: dict[int, dict] = {}
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "step" not in record or "loss" not in record:
                continue
            by_step[int(record["step"])] = record
    if not by_step:
        raise ValueError(f"no OPD training records found in {path}")
    return [by_step[step] for step in sorted(by_step)]


def values(records: list[dict], key: str) -> np.ndarray:
    return np.asarray([float(record[key]) for record in records], dtype=np.float64)


def ema(series: np.ndarray, span: int) -> np.ndarray:
    alpha = 2.0 / (max(span, 1) + 1.0)
    result = np.empty_like(series)
    result[0] = series[0]
    for index in range(1, len(series)):
        result[index] = alpha * series[index] + (1.0 - alpha) * result[index - 1]
    return result


def main() -> None:
    args = parse_args()
    if args.window < 1:
        raise ValueError("--window must be positive")

    records = load_records(args.log)
    steps = values(records, "step")
    total_loss = values(records, "loss")
    first_kl = values(records, "first_kl")
    sub_kl = values(records, "sub_kl") * 0.3
    student_ce = values(records, "student_ce") * 0.05
    total_steps = int(records[-1].get("total_steps", steps[-1]))
    progress = 100.0 * steps[-1] / max(total_steps, 1)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titleweight": "bold",
            "axes.edgecolor": "#98A2B3",
            "axes.labelcolor": "#344054",
            "xtick.color": "#475467",
            "ytick.color": "#475467",
            "figure.facecolor": "#F8FAFC",
        }
    )
    figure, (loss_axis, component_axis) = plt.subplots(
        2,
        1,
        figsize=(12, 7.2),
        dpi=160,
        sharex=True,
        gridspec_kw={"height_ratios": [1.65, 1.0], "hspace": 0.13},
    )

    for axis in (loss_axis, component_axis):
        axis.set_facecolor("#FFFFFF")
        axis.grid(axis="y", color="#E4E7EC", linewidth=0.8)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    loss_axis.plot(steps, total_loss, color="#98A2B3", alpha=0.30, linewidth=0.75, label="Raw loss")
    smoothed_loss = ema(total_loss, args.window)
    loss_axis.plot(
        steps,
        smoothed_loss,
        color="#087E8B",
        linewidth=2.2,
        label=f"Loss EMA ({args.window} steps)",
    )
    loss_axis.set_ylabel("Loss")
    loss_axis.set_title("EmotionTalk OPD Training Loss", loc="left", fontsize=16, pad=24)
    loss_axis.text(
        0.0,
        1.035,
        f"Step {int(steps[-1]):,} / {total_steps:,} ({progress:.1f}%)  |  "
        f"latest {total_loss[-1]:.4f}  |  EMA {smoothed_loss[-1]:.4f}",
        transform=loss_axis.transAxes,
        color="#475467",
        fontsize=9.5,
    )
    loss_axis.legend(frameon=False, loc="upper right", ncol=2)

    component_axis.plot(steps, ema(first_kl, args.window), color="#D1495B", linewidth=1.8, label="First KL")
    component_axis.plot(steps, ema(sub_kl, args.window), color="#3066BE", linewidth=1.8, label="0.30 x Sub KL")
    component_axis.plot(
        steps,
        ema(student_ce, args.window),
        color="#E08E0B",
        linewidth=1.8,
        label="0.05 x Student CE",
    )
    component_axis.set_ylabel("Weighted contribution")
    component_axis.set_xlabel("Training step")
    component_axis.legend(frameon=False, loc="upper right", ncol=3)

    finish_at = records[-1].get("finish_at")
    generated_at = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")
    footer = f"Source: {args.log}  |  generated {generated_at}"
    if finish_at:
        footer += f"  |  current ETA finish {finish_at}"
    figure.text(0.075, 0.012, footer, color="#667085", fontsize=8)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = args.output.with_name(f".{args.output.name}.tmp")
    figure.savefig(temporary_output, format="png", bbox_inches="tight", facecolor=figure.get_facecolor())
    plt.close(figure)
    os.replace(temporary_output, args.output)
    print(
        json.dumps(
            {
                "records": len(records),
                "step": int(steps[-1]),
                "total_steps": total_steps,
                "progress_percent": round(progress, 2),
                "latest_loss": float(total_loss[-1]),
                "ema_loss": float(smoothed_loss[-1]),
                "output": str(args.output),
            }
        )
    )


if __name__ == "__main__":
    main()
