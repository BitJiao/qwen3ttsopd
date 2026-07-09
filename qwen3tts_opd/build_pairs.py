from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from qwen3tts_opd.core import (
    call_rewards,
    ensure_qwen3_tts_repo_on_path,
    generate_voice_clone_rollouts,
    import_reward_fn,
    load_jsonl,
    load_tts,
    resolve_local_model_dir,
    torch_dtype,
)
from qwen3tts_opd.instruction_utils import get_target_text, with_formatted_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build offline OPD/DPO preference pairs for Qwen3-TTS.")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--audio_dir", default=None)
    parser.add_argument("--reward_fn", default="qwen3tts_opd.reward.wer_sim_reward:compute_score")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--attn_implementation", default="sdpa")
    parser.add_argument("--group_size", type=int, default=4)
    parser.add_argument("--teacher_icl", action="store_true", default=True)
    parser.add_argument("--teacher_xvec", dest="teacher_icl", action="store_false")
    parser.add_argument("--include_teacher_as_candidate", action="store_true", default=True)
    parser.add_argument("--student_xvec_only", action="store_true", default=True)
    parser.add_argument("--instruction_template", default="qwen_control", choices=["qwen_control", "plain", "bracket"])
    parser.add_argument("--margin", type=float, default=0.05)
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--subtalker_temperature", type=float, default=0.9)
    parser.add_argument("--subtalker_top_k", type=int, default=50)
    parser.add_argument("--subtalker_top_p", type=float, default=1.0)
    parser.add_argument("--non_streaming_mode", action="store_true", default=True)
    parser.add_argument("--streaming_mode", dest="non_streaming_mode", action="store_false")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _write_wav(path: Path, wav: np.ndarray, sample_rate: int) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), wav, sample_rate)
    return str(path)


@torch.no_grad()
def generate_teacher_icl(tts, sample: dict, args):
    old_mode = args.x_vector_only_mode
    try:
        args.x_vector_only_mode = not args.teacher_icl
        codes, wavs, sample_rate = generate_voice_clone_rollouts(tts, sample, 1, args)
    finally:
        args.x_vector_only_mode = old_mode
    return codes[0], wavs[0], sample_rate


def main() -> None:
    ensure_qwen3_tts_repo_on_path()
    args = parse_args()
    args.x_vector_only_mode = args.student_xvec_only
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.group_size < 2:
        raise ValueError("--group_size must be >= 2")
    if Path(args.output_jsonl).exists() and not args.overwrite:
        raise FileExistsError(f"{args.output_jsonl} exists; pass --overwrite")

    local_model_dir = resolve_local_model_dir(args.model_path)
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    dtype = torch_dtype(args.dtype if device.type != "cpu" else "fp32")
    tts = load_tts(local_model_dir, dtype, args.attn_implementation, device)
    tts.model.eval()

    data = load_jsonl(args.input_jsonl)
    if args.max_samples > 0:
        data = data[: args.max_samples]
    reward_fn = import_reward_fn(args.reward_fn)

    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    audio_dir = Path(args.audio_dir) if args.audio_dir else output_path.with_suffix("")

    kept = 0
    skipped = 0
    with output_path.open("w", encoding="utf-8") as out:
        for idx, raw_sample in enumerate(data):
            generation_sample = with_formatted_text(raw_sample, template=args.instruction_template)
            reward_sample = dict(raw_sample)
            reward_sample["text"] = get_target_text(raw_sample)
            if "ref_audio" not in generation_sample:
                raise KeyError("each row must contain ref_audio")

            candidates = []
            if args.include_teacher_as_candidate:
                teacher_codes, teacher_wav, sample_rate = generate_teacher_icl(tts, generation_sample, args)
                candidates.append(
                    {
                        "source": "teacher_icl" if args.teacher_icl else "teacher_xvec",
                        "codes": teacher_codes,
                        "wav": teacher_wav,
                    }
                )

            old_mode = args.x_vector_only_mode
            try:
                args.x_vector_only_mode = args.student_xvec_only
                codes_list, wavs, sample_rate = generate_voice_clone_rollouts(
                    tts,
                    generation_sample,
                    args.group_size,
                    args,
                )
            finally:
                args.x_vector_only_mode = old_mode

            for cand_idx, (codes, wav) in enumerate(zip(codes_list, wavs)):
                candidates.append({"source": f"student_{cand_idx}", "codes": codes, "wav": wav})

            scores = call_rewards(
                reward_fn,
                sample=reward_sample,
                wavs=[c["wav"] for c in candidates],
                sample_rate=sample_rate,
                codes_list=[c["codes"] for c in candidates],
            )
            for candidate, score in zip(candidates, scores):
                candidate["score"] = float(score)

            ordered = sorted(candidates, key=lambda c: c["score"], reverse=True)
            chosen = ordered[0]
            rejected = ordered[-1]
            if chosen["score"] - rejected["score"] < args.margin:
                skipped += 1
                continue

            prefix = f"{idx:08d}"
            chosen_audio = _write_wav(audio_dir / f"{prefix}_chosen.wav", chosen["wav"], sample_rate)
            rejected_audio = _write_wav(audio_dir / f"{prefix}_rejected.wav", rejected["wav"], sample_rate)

            row = {
                "sample_id": raw_sample.get("sample_id", idx),
                "text": generation_sample["text"],
                "raw_text": generation_sample.get("raw_text", raw_sample.get("text")),
                "instruction": raw_sample.get("instruction")
                or raw_sample.get("emotion_instruction")
                or raw_sample.get("style_instruction"),
                "language": generation_sample.get("language", "Auto"),
                "ref_audio": generation_sample["ref_audio"],
                "ref_text": generation_sample.get("ref_text"),
                "chosen_audio": chosen_audio,
                "rejected_audio": rejected_audio,
                "chosen_codes": chosen["codes"].detach().cpu().tolist(),
                "rejected_codes": rejected["codes"].detach().cpu().tolist(),
                "chosen_score": chosen["score"],
                "rejected_score": rejected["score"],
                "chosen_source": chosen["source"],
                "rejected_source": rejected["source"],
                "candidate_scores": [{"source": c["source"], "score": c["score"]} for c in candidates],
            }
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            kept += 1

            print(
                json.dumps(
                    {
                        "idx": idx,
                        "kept": kept,
                        "skipped": skipped,
                        "chosen": row["chosen_source"],
                        "rejected": row["rejected_source"],
                        "chosen_score": row["chosen_score"],
                        "rejected_score": row["rejected_score"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    print(json.dumps({"pairs": kept, "skipped": skipped, "output": str(output_path)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

