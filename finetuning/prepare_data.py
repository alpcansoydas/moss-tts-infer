from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Optional

import torch
import torchaudio
from accelerate import Accelerator
from transformers import AutoModel

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from finetuning.common import (
    dump_jsonl,
    format_timestamp,
    load_jsonl,
    normalize_audio_path_list,
    resolve_record_audio_paths,
    resolve_shard_spec,
    select_rank_shard,
    shard_output_path,
)

DEFAULT_CODEC_PATH = REPO_ROOT / "models" / "MOSS-Audio-Tokenizer-Nano"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute audio codes for MOSS-TTS-Nano finetuning.")
    parser.add_argument("--codec-path", type=str, default=str(DEFAULT_CODEC_PATH))
    parser.add_argument("--device", type=str, default="auto", help="Codec device. Use `auto` to follow Accelerate rank.")
    parser.add_argument("--input-jsonl", type=str, required=True)
    parser.add_argument("--output-jsonl", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--n-vq", type=int, default=None, help="Only keep the first n_vq codec layers.")
    parser.add_argument("--num-shards", type=int, default=None)
    parser.add_argument("--shard-rank", type=int, default=None)
    parser.add_argument(
        "--skip-reference-audio-codes",
        dest="encode_reference_audio",
        action="store_false",
        help="Only encode `audio_codes`. Skip `ref_audio_codes`.",
    )
    parser.add_argument("--save-shard-suffix", action="store_true")
    parser.set_defaults(encode_reference_audio=True)
    return parser.parse_args()


def load_codec(codec_path: str, device: str):
    codec = AutoModel.from_pretrained(codec_path, trust_remote_code=True)
    if hasattr(codec, "set_attention_implementation"):
        try:
            codec.set_attention_implementation("sdpa" if "cuda" in str(device) else "sdpa")
        except Exception:
            pass
    if hasattr(codec, "eval"):
        codec.eval()
    if hasattr(codec, "to"):
        codec = codec.to(device)
    return codec


def resolve_codec_sample_rate(codec) -> int:
    for holder in (codec, getattr(codec, "config", None)):
        for attr_name in ("sampling_rate", "sample_rate"):
            value = getattr(holder, attr_name, None)
            if value is not None:
                return int(value)
    raise ValueError("Codec sample rate is missing.")


def resolve_codec_channels(codec) -> int:
    for holder in (codec, getattr(codec, "config", None)):
        for attr_name in ("number_channels", "channels_numbers"):
            value = getattr(holder, attr_name, None)
            if value is not None:
                return int(value)
    raise ValueError("Codec target channel count is missing.")


def resolve_codec_device(codec) -> torch.device:
    if hasattr(codec, "device"):
        try:
            return torch.device(codec.device)
        except Exception:
            pass
    try:
        return next(codec.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def load_audio_for_codec(path: str, codec) -> torch.Tensor:
    waveform, sample_rate = torchaudio.load(path)
    waveform = waveform.to(torch.float32)

    target_sample_rate = resolve_codec_sample_rate(codec)
    if sample_rate != target_sample_rate:
        waveform = torchaudio.functional.resample(waveform, sample_rate, target_sample_rate)

    target_channels = resolve_codec_channels(codec)
    if waveform.shape[0] == target_channels:
        return waveform.contiguous()
    if waveform.shape[0] == 1 and target_channels > 1:
        return waveform.repeat(target_channels, 1).contiguous()
    if waveform.shape[0] > 1 and target_channels == 1:
        return waveform.mean(dim=0, keepdim=True).contiguous()
    if waveform.shape[0] > target_channels:
        return waveform[:target_channels].contiguous()
    raise ValueError(f"Unsupported channel conversion for {path}: {waveform.shape[0]} -> {target_channels}")


def encode_audio_paths(
    codec,
    paths: Iterable[str],
    *,
    batch_size: int,
    n_vq: Optional[int],
) -> Dict[str, List[List[int]]]:
    unique_paths = list(dict.fromkeys(str(path) for path in paths))
    outputs: Dict[str, List[List[int]]] = {}
    if not unique_paths:
        return outputs

    codec_device = resolve_codec_device(codec)
    for start in range(0, len(unique_paths), batch_size):
        batch_paths = unique_paths[start : start + batch_size]
        wav_list = [load_audio_for_codec(path, codec).to(codec_device) for path in batch_paths]
        encoded = codec.batch_encode(wav_list, num_quantizers=n_vq)
        audio_codes = encoded.audio_codes
        audio_code_lengths = encoded.audio_codes_lengths
        if audio_codes is None or audio_code_lengths is None:
            raise RuntimeError("Codec batch_encode() did not return audio_codes/audio_codes_lengths.")
        for batch_index, path in enumerate(batch_paths):
            length = int(audio_code_lengths[batch_index].item())
            sample_codes = audio_codes[:, batch_index, :length].transpose(0, 1).contiguous()
            outputs[path] = sample_codes.cpu().tolist()
    return outputs


def collect_missing_reference_paths(records: List[Dict[str, Any]]) -> List[str]:
    paths: List[str] = []
    for record in records:
        if record.get("ref_audio_codes") is not None:
            continue
        values = normalize_audio_path_list(record.get("ref_audio"), "ref_audio")
        if values is not None:
            paths.extend(values)
    return list(dict.fromkeys(paths))


def attach_reference_audio_codes(
    records: List[Dict[str, Any]],
    *,
    encoded_reference_paths: Dict[str, List[List[int]]],
) -> None:
    for record in records:
        if record.get("ref_audio_codes") is not None:
            continue
        ref_audio = normalize_audio_path_list(record.get("ref_audio"), "ref_audio")
        if ref_audio is None:
            continue
        if len(ref_audio) != 1:
            raise ValueError("`ref_audio` only supports a single path.")
        record["ref_audio_codes"] = encoded_reference_paths[ref_audio[0]]


def main() -> None:
    args = parse_args()
    accelerator = Accelerator()
    device = str(accelerator.device) if args.device == "auto" else args.device

    input_jsonl_path = Path(args.input_jsonl).resolve()
    all_records = [
        resolve_record_audio_paths(record, base_dir=input_jsonl_path.parent)
        for record in load_jsonl(input_jsonl_path)
    ]
    world_size, rank = resolve_shard_spec(
        args.num_shards,
        args.shard_rank,
        default_num_shards=accelerator.num_processes,
        default_shard_rank=accelerator.process_index,
    )
    records = select_rank_shard(all_records, world_size, rank)
    if not records:
        raise ValueError(
            f"No records found for shard rank={rank} / world_size={world_size} in {input_jsonl_path}."
        )

    codec = load_codec(args.codec_path, device=device)

    target_paths: List[str] = []
    for index, record in enumerate(records):
        if record.get("audio_codes") is not None:
            continue
        audio_path = record.get("audio")
        if not isinstance(audio_path, str) or not audio_path:
            raise ValueError(f"Record {index} is missing a valid `audio` field.")
        target_paths.append(audio_path)

    encoded_target_paths = encode_audio_paths(
        codec,
        target_paths,
        batch_size=args.batch_size,
        n_vq=args.n_vq,
    )
    for record in records:
        if record.get("audio_codes") is None:
            record["audio_codes"] = encoded_target_paths[str(record["audio"])]

    if args.encode_reference_audio:
        reference_paths = collect_missing_reference_paths(records)
        encoded_reference_paths = encode_audio_paths(
            codec,
            reference_paths,
            batch_size=args.batch_size,
            n_vq=args.n_vq,
        )
        attach_reference_audio_codes(records, encoded_reference_paths=encoded_reference_paths)

    output_path = Path(args.output_jsonl)
    if world_size > 1 or args.save_shard_suffix:
        output_path = shard_output_path(output_path, rank, world_size)
    dump_jsonl(records, output_path)
    accelerator.print(
        f"[{format_timestamp()}] [prepare_data] rank={rank}/{world_size} "
        f"records={len(records)} device={device} output={output_path}"
    )


if __name__ == "__main__":
    main()
