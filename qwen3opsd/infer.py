from __future__ import annotations

import argparse
from pathlib import Path

import soundfile as sf
import torch


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Base voice-clone or VoiceDesign inference with an SFT/OPD checkpoint."
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--student-spk-audio", default=None)
    parser.add_argument("--text", required=True)
    parser.add_argument("--instruction", default="")
    parser.add_argument("--language", default="Chinese")
    parser.add_argument("--output-wav", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    args = parser.parse_args()

    from qwen_tts import Qwen3TTSModel
    from qwen3tts_opd.core import generate_instructed_voice_clone

    model = Qwen3TTSModel.from_pretrained(
        args.model_path,
        device_map=args.device,
        dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
    )
    if model.model.tts_model_type == "voice_design":
        wavs, sample_rate = model.generate_voice_design(
            text=args.text,
            instruct=args.instruction,
            language=args.language,
            non_streaming_mode=True,
            max_new_tokens=args.max_new_tokens,
        )
    else:
        if not args.student_spk_audio:
            parser.error("Base checkpoints require --student-spk-audio")
        prompt_items = model.create_voice_clone_prompt(
            ref_audio=args.student_spk_audio,
            ref_text=None,
            x_vector_only_mode=True,
        )
        prompt = model._prompt_items_to_voice_clone_prompt(prompt_items)
        wavs, sample_rate = generate_instructed_voice_clone(
            model,
            text=args.text,
            instruction=args.instruction,
            language=args.language,
            voice_clone_prompt=prompt,
            non_streaming_mode=True,
            max_new_tokens=args.max_new_tokens,
        )
    args.output_wav.parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.output_wav, wavs[0], sample_rate)


if __name__ == "__main__":
    main()
