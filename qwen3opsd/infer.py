from __future__ import annotations

import argparse
from pathlib import Path

import soundfile as sf
import torch

from qwen3opsd.instruction_utils import format_instruction_text


def main() -> None:
    parser = argparse.ArgumentParser(description="Run instruction-conditioned x-vector inference with an SFT/OPD checkpoint.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--student-spk-audio", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--instruction", default="")
    parser.add_argument("--output-wav", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    args = parser.parse_args()

    from qwen_tts import Qwen3TTSModel

    model = Qwen3TTSModel.from_pretrained(
        args.model_path,
        device_map=args.device,
        dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
    )
    conditioned_text = format_instruction_text(
        {"text": args.text, "instruction": args.instruction},
        template="qwen_control",
    )
    wavs, sample_rate = model.generate_voice_clone(
        text=conditioned_text,
        language="Chinese",
        ref_audio=args.student_spk_audio,
        x_vector_only_mode=True,
        non_streaming_mode=True,
        max_new_tokens=args.max_new_tokens,
    )
    args.output_wav.parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.output_wav, wavs[0], sample_rate)


if __name__ == "__main__":
    main()
