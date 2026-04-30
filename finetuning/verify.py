from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

import torch
from transformers import AutoModelForCausalLM


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT_PATH = REPO_ROOT / "models" / "MOSS-TTS-Nano"
DEFAULT_CODEC_PATH = REPO_ROOT / "models" / "MOSS-Audio-Tokenizer-Nano"
DEFAULT_OUTPUT_AUDIO_PATH = REPO_ROOT / "generated_audio" / "finetune_verify.wav"
MOSS_AUDIO_TOKENIZER_TYPE = "moss-audio-tokenizer-nano"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quick non-streaming validation for MOSS-TTS-Nano finetune checkpoints.")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT_PATH))
    parser.add_argument("--output-audio-path", default=str(DEFAULT_OUTPUT_AUDIO_PATH))

    text_group = parser.add_mutually_exclusive_group(required=True)
    text_group.add_argument("--text", help="Text to synthesize.")
    text_group.add_argument("--text-file", help="UTF-8 text file to synthesize.")

    prompt_text_group = parser.add_mutually_exclusive_group(required=False)
    prompt_text_group.add_argument("--prompt-text", help="Prompt transcript for continuation mode.")
    prompt_text_group.add_argument("--prompt-text-file", help="UTF-8 prompt transcript file.")

    parser.add_argument("--mode", default="voice_clone", choices=("continuation", "voice_clone"))
    parser.add_argument("--prompt-audio-path", default=None)
    parser.add_argument("--reference-audio-path", default=None, help="Compatibility alias for --prompt-audio-path.")
    parser.add_argument(
        "--audio-tokenizer-pretrained-name-or-path",
        default=str(DEFAULT_CODEC_PATH),
        help="Local codec path or HF repo id. Defaults to repo-local models/MOSS-Audio-Tokenizer-Nano.",
    )
    parser.add_argument("--text-tokenizer-path", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto", choices=("auto", "float32", "float16", "bfloat16"))
    parser.add_argument("--nq", type=int, default=None)
    parser.add_argument("--max-new-frames", type=int, default=375)
    parser.add_argument("--do-sample", type=int, nargs="?", const=1, default=1, choices=[0, 1])
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args(argv)


def resolve_text(args: argparse.Namespace) -> str:
    if args.text is not None:
        return args.text
    return Path(args.text_file).read_text(encoding="utf-8")


def resolve_prompt_text(args: argparse.Namespace) -> Optional[str]:
    if args.prompt_text is not None:
        return args.prompt_text
    if args.prompt_text_file is not None:
        return Path(args.prompt_text_file).read_text(encoding="utf-8")
    return None


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def resolve_dtype(dtype_arg: str, device: torch.device) -> torch.dtype:
    if dtype_arg == "float32":
        return torch.float32
    if dtype_arg == "float16":
        return torch.float16
    if dtype_arg == "bfloat16":
        return torch.bfloat16
    if device.type == "cuda":
        if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def main(argv: Optional[Sequence[str]] = None) -> dict[str, object]:
    args = parse_args(argv)
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    model = AutoModelForCausalLM.from_pretrained(args.checkpoint, trust_remote_code=True)
    model.to(device=device, dtype=dtype)
    if hasattr(model, "_set_attention_implementation"):
        model._set_attention_implementation("sdpa" if device.type == "cuda" else "eager")
    model.eval()

    text = resolve_text(args)
    prompt_text = resolve_prompt_text(args)
    result = model.inference(
        text=text,
        output_audio_path=args.output_audio_path,
        mode=args.mode,
        prompt_text=prompt_text,
        prompt_audio_path=args.prompt_audio_path,
        reference_audio_path=args.reference_audio_path,
        text_tokenizer_path=args.text_tokenizer_path or args.checkpoint,
        audio_tokenizer_type=MOSS_AUDIO_TOKENIZER_TYPE,
        audio_tokenizer_pretrained_name_or_path=args.audio_tokenizer_pretrained_name_or_path,
        device=device,
        nq=args.nq,
        max_new_frames=args.max_new_frames,
        do_sample=bool(args.do_sample),
        use_kv_cache=True,
    )
    print(
        f"saved {result['audio_path']} sample_rate={result['sample_rate']} "
        f"frames={int(result['audio_token_ids'].shape[0])}"
    )
    return result


if __name__ == "__main__":
    main()
