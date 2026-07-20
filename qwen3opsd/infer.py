from __future__ import annotations

import argparse
from pathlib import Path

import soundfile as sf
import torch

def main() -> None:
    parser = argparse.ArgumentParser(description="Run Qwen3-TTS VoiceDesign inference with an SFT/OPD checkpoint.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--instruction", default="")
    parser.add_argument("--language", default="Chinese")
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
    wavs, sample_rate = model.generate_voice_design(
        text=args.text,
        instruct=args.instruction,
        language=args.language,
        non_streaming_mode=True,
        max_new_tokens=args.max_new_tokens,
    )
    args.output_wav.parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.output_wav, wavs[0], sample_rate)


if __name__ == "__main__":
    main()
