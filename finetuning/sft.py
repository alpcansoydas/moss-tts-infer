from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed
from accelerate.utils.dataclasses import DistributedDataParallelKwargs
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler
from transformers.utils import cached_file

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from finetuning.common import format_duration, format_timestamp, load_jsonl_spec
from finetuning.dataset import MossTTSNanoSFTDataset

DEFAULT_MODEL_PATH = REPO_ROOT / "models" / "MOSS-TTS-Nano"
DEFAULT_CODEC_PATH = REPO_ROOT / "models" / "MOSS-Audio-Tokenizer-Nano"

SCHEDULER_CHOICES = (
    "linear",
    "cosine",
    "cosine_with_restarts",
    "polynomial",
    "constant",
    "constant_with_warmup",
    "inverse_sqrt",
)

MODEL_SUPPORT_FILES = (
    "__init__.py",
    "configuration_moss_tts_nano.py",
    "gpt2_decoder.py",
    "modeling_moss_tts_nano.py",
    "prompting.py",
    "tokenization_moss_tts_nano.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple supervised finetuning for MOSS-TTS-Nano.")
    parser.add_argument("--model-path", type=str, default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--codec-path", type=str, default=str(DEFAULT_CODEC_PATH))
    parser.add_argument(
        "--train-jsonl",
        type=str,
        required=True,
        help="A single JSONL, directory, glob, or comma-separated list of JSONLs produced by prepare_data.py.",
    )
    parser.add_argument("--output-dir", type=str, default="output/moss_tts_nano_sft")
    parser.add_argument("--max-length", type=int, default=1024, help="Fixed full sequence length before shift.")
    parser.add_argument("--per-device-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.95)
    parser.add_argument("--adam-eps", type=float, default=1e-8)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--lr-scheduler-type", type=str, default="linear", choices=SCHEDULER_CHOICES)
    parser.add_argument("--num-epochs", type=int, default=3)
    parser.add_argument("--max-train-steps", type=int, default=None)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-every-epochs", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--mixed-precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--attn-implementation", type=str, default="auto")
    parser.add_argument(
        "--channelwise-loss-weight",
        type=str,
        default="1,32",
        help=(
            "Either n_heads values (text,vq0,...,vqN) or 2 values (text_weight,total_audio_weight). "
            "The total audio weight will be evenly split across all audio heads."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.max_length <= 8:
        raise ValueError("`max_length` must be > 8.")
    if args.per_device_batch_size <= 0:
        raise ValueError("`per_device_batch_size` must be > 0.")
    if args.gradient_accumulation_steps <= 0:
        raise ValueError("`gradient_accumulation_steps` must be > 0.")
    if args.learning_rate <= 0:
        raise ValueError("`learning_rate` must be > 0.")
    if args.weight_decay < 0:
        raise ValueError("`weight_decay` must be >= 0.")
    if args.warmup_steps < 0:
        raise ValueError("`warmup_steps` must be >= 0.")
    if not 0.0 <= args.warmup_ratio < 1.0:
        raise ValueError("`warmup_ratio` must be in [0, 1).")
    if args.num_epochs <= 0:
        raise ValueError("`num_epochs` must be > 0.")
    if args.max_train_steps is not None and args.max_train_steps <= 0:
        raise ValueError("`max_train_steps` must be > 0 when set.")
    if args.max_grad_norm < 0:
        raise ValueError("`max_grad_norm` must be >= 0.")
    if args.logging_steps <= 0:
        raise ValueError("`logging_steps` must be > 0.")
    if args.save_every_epochs <= 0:
        raise ValueError("`save_every_epochs` must be > 0.")
    if args.num_workers < 0:
        raise ValueError("`num_workers` must be >= 0.")


def configure_torch_backends() -> None:
    if not torch.cuda.is_available():
        return
    if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
        torch.backends.cuda.enable_cudnn_sdp(False)
    if hasattr(torch.backends.cuda, "enable_flash_sdp"):
        torch.backends.cuda.enable_flash_sdp(True)
    if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
        torch.backends.cuda.enable_mem_efficient_sdp(True)
    if hasattr(torch.backends.cuda, "enable_math_sdp"):
        torch.backends.cuda.enable_math_sdp(True)


def resolve_torch_dtype(mixed_precision: str) -> torch.dtype:
    if not torch.cuda.is_available():
        return torch.float32
    if mixed_precision == "fp16":
        return torch.float16
    if mixed_precision == "bf16":
        return torch.bfloat16
    return torch.float32


def resolve_accelerate_mixed_precision(mixed_precision: str) -> str:
    if not torch.cuda.is_available():
        return "no"
    return mixed_precision


def resolve_attn_implementation(requested: str, dtype: torch.dtype) -> str:
    if requested != "auto":
        return requested
    if not torch.cuda.is_available():
        return "eager"
    if dtype in {torch.float16, torch.bfloat16}:
        try:
            import flash_attn  # noqa: F401

            major, _ = torch.cuda.get_device_capability()
            if major >= 8:
                return "flash_attention_2"
        except Exception:
            pass
    return "sdpa"


def resolve_warmup_steps(args: argparse.Namespace, num_training_steps: int) -> int:
    if args.warmup_steps > 0:
        return args.warmup_steps
    if args.warmup_ratio > 0:
        return math.ceil(num_training_steps * args.warmup_ratio)
    return 0


def parse_channelwise_loss_weight(spec: str, n_heads: int) -> List[float]:
    values = [float(item.strip()) for item in str(spec).split(",") if item.strip()]
    if len(values) == n_heads:
        resolved = values
    elif len(values) == 2 and n_heads > 1:
        text_weight, total_audio_weight = values
        per_audio_weight = total_audio_weight / float(n_heads - 1)
        resolved = [text_weight] + [per_audio_weight] * (n_heads - 1)
    else:
        raise ValueError(
            f"`channelwise_loss_weight` expects either {n_heads} values or 2 values, got {len(values)}."
        )
    if sum(resolved) <= 0:
        raise ValueError("`channelwise_loss_weight` must sum to a positive value.")
    return resolved


def build_optimizer(model, args: argparse.Namespace) -> AdamW:
    return AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_eps,
    )


def unwrap_training_model(model):
    unwrapped = model
    while hasattr(unwrapped, "module"):
        unwrapped = unwrapped.module
    return unwrapped


def compute_supervised_loss(
    model,
    *,
    input_ids: torch.LongTensor,
    attention_mask: torch.BoolTensor,
    labels: torch.LongTensor,
    channelwise_loss_weight: List[float],
) -> torch.Tensor:
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
        return_dict=True,
    )
    global_hidden_states = outputs.global_hidden_states
    if global_hidden_states is None:
        raise RuntimeError("Model forward did not return global_hidden_states.")

    base_model = unwrap_training_model(model)
    batch_size, seq_len, hidden_size = global_hidden_states.shape
    n_vq = int(base_model.config.n_vq)
    flat_hidden = global_hidden_states.reshape(batch_size * seq_len, hidden_size)
    local_dtype = base_model.local_transformer.ln_f.weight.dtype
    flat_hidden = flat_hidden.to(dtype=local_dtype)

    flat_labels = labels.reshape(batch_size * seq_len, n_vq + 1)
    local_inputs = torch.zeros(
        (batch_size * seq_len, n_vq + 1, hidden_size),
        dtype=local_dtype,
        device=flat_hidden.device,
    )
    local_inputs[:, 0, :] = flat_hidden

    text_targets = flat_labels[:, 0]
    safe_text_targets = text_targets.masked_fill(text_targets.lt(0), int(base_model.config.pad_token_id))
    local_inputs[:, 1, :] = base_model.transformer.wte(safe_text_targets)

    audio_targets = flat_labels[:, 1:]
    for channel_index in range(n_vq - 1):
        teacher_ids = audio_targets[:, channel_index]
        valid_mask = (teacher_ids >= 0) & (teacher_ids < base_model.audio_embeddings[channel_index].num_embeddings)
        safe_ids = teacher_ids.masked_fill(~valid_mask, 0)
        channel_embeds = base_model.audio_embeddings[channel_index](safe_ids)
        channel_embeds = channel_embeds * valid_mask.unsqueeze(-1)
        local_inputs[:, channel_index + 2, :] = channel_embeds.to(dtype=local_dtype)

    local_attention_mask = torch.ones(
        (batch_size * seq_len, n_vq + 1),
        dtype=torch.bool,
        device=flat_hidden.device,
    )
    local_outputs = base_model.local_transformer(
        input_ids=None,
        attention_mask=local_attention_mask,
        position_ids=None,
        inputs_embeds=local_inputs,
        use_cache=False,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
        cu_seqlens=None,
        num_sequences=None,
    )
    local_hidden_states = local_outputs.last_hidden_state

    total_loss = torch.zeros((), device=flat_hidden.device, dtype=torch.float32)
    total_weight = 0.0

    text_logits = base_model.text_lm_head(local_hidden_states[:, 0, :])
    if (text_targets != -100).any():
        text_loss = F.cross_entropy(text_logits.float(), text_targets, ignore_index=-100)
        total_loss = total_loss + float(channelwise_loss_weight[0]) * text_loss.float()
        total_weight += float(channelwise_loss_weight[0])

    for channel_index in range(n_vq):
        channel_targets = audio_targets[:, channel_index]
        if not (channel_targets != -100).any():
            continue
        channel_logits = base_model.audio_lm_heads[channel_index](local_hidden_states[:, channel_index + 1, :])
        channel_loss = F.cross_entropy(channel_logits.float(), channel_targets, ignore_index=-100)
        total_loss = total_loss + float(channelwise_loss_weight[channel_index + 1]) * channel_loss.float()
        total_weight += float(channelwise_loss_weight[channel_index + 1])

    if total_weight <= 0:
        raise RuntimeError("All labels are ignored; check dataset packing and max_length.")
    return total_loss / total_weight


def resolve_asset(model_path: str, filename: str) -> Optional[Path]:
    model_path_obj = Path(model_path)
    if model_path_obj.is_dir():
        candidate = model_path_obj / filename
        return candidate if candidate.exists() else None

    try:
        resolved = cached_file(
            model_path,
            filename,
            _raise_exceptions_for_missing_entries=False,
        )
    except OSError:
        return None

    if resolved is None:
        return None
    return Path(resolved)


def save_checkpoint(
    *,
    accelerator: Accelerator,
    model,
    tokenizer,
    model_path: str,
    codec_path: str,
    output_dir: Path,
    train_args: Dict[str, Any],
    global_step: int,
    epoch: int,
) -> None:
    accelerator.wait_for_everyone()
    if not accelerator.is_main_process:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    unwrapped_model = unwrap_training_model(model)
    unwrapped_model.config.audio_tokenizer_pretrained_name_or_path = str(Path(codec_path).expanduser().resolve())
    unwrapped_model.config.save_pretrained(output_dir)
    state_dict = {
        key: value.detach().cpu()
        for key, value in unwrapped_model.state_dict().items()
    }
    torch.save(state_dict, output_dir / "pytorch_model.bin")
    tokenizer.save_pretrained(output_dir)

    for filename in MODEL_SUPPORT_FILES:
        src = resolve_asset(model_path, filename)
        if src is not None and src.exists():
            shutil.copy2(src, output_dir / filename)

    metadata = dict(train_args)
    metadata["saved_global_step"] = int(global_step)
    metadata["saved_epoch"] = int(epoch)
    metadata["saved_at"] = format_timestamp()
    metadata["checkpoint_dir"] = str(output_dir)
    with open(output_dir / "finetune_config.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)


def main() -> None:
    args = parse_args()
    validate_args(args)
    configure_torch_backends()
    set_seed(args.seed)

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=resolve_accelerate_mixed_precision(args.mixed_precision),
        step_scheduler_with_optimizer=False,
        kwargs_handlers=[ddp_kwargs],
    )
    if accelerator.device.type != "cuda":
        raise EnvironmentError(
            f"MOSS-TTS-Nano finetuning requires CUDA, but Accelerate resolved device={accelerator.device}."
        )

    model_dtype = resolve_torch_dtype(args.mixed_precision)
    attn_implementation = resolve_attn_implementation(args.attn_implementation, model_dtype)
    records_paths, records = load_jsonl_spec(args.train_jsonl)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=model_dtype,
    )
    if hasattr(model, "_set_attention_implementation"):
        model._set_attention_implementation(attn_implementation)

    dataset = MossTTSNanoSFTDataset(
        records,
        tokenizer=tokenizer,
        model_config=model.config,
        max_length=args.max_length,
    )
    train_dataloader = DataLoader(
        dataset,
        batch_size=args.per_device_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=dataset.collate_fn,
    )

    optimizer = build_optimizer(model, args)
    global_batch_size = (
        args.per_device_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    )
    micro_batches_per_epoch = math.ceil(len(dataset) / (args.per_device_batch_size * accelerator.num_processes))
    optimizer_steps_per_epoch = math.ceil(micro_batches_per_epoch / args.gradient_accumulation_steps)
    max_train_steps = args.max_train_steps or (args.num_epochs * optimizer_steps_per_epoch)
    warmup_steps = resolve_warmup_steps(args, max_train_steps)
    channelwise_loss_weight = parse_channelwise_loss_weight(
        args.channelwise_loss_weight,
        int(model.config.n_vq) + 1,
    )

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max_train_steps,
    )
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model,
        optimizer,
        train_dataloader,
        lr_scheduler,
    )

    output_root = Path(args.output_dir)
    if accelerator.is_main_process:
        output_root.mkdir(parents=True, exist_ok=True)

    train_args_to_save = vars(args).copy()
    train_args_to_save["resolved_warmup_steps"] = warmup_steps
    train_args_to_save["resolved_channelwise_loss_weight"] = channelwise_loss_weight
    train_args_to_save["global_batch_size"] = global_batch_size
    train_args_to_save["records_paths"] = [str(path.resolve()) for path in records_paths]
    train_args_to_save["attn_implementation"] = attn_implementation

    accelerator.print(
        f"[{format_timestamp()}] [sft] loaded_records={len(dataset)} "
        f"device={accelerator.device} "
        f"num_processes={accelerator.num_processes} "
        f"global_batch_size={global_batch_size} "
        f"micro_batches_per_epoch={micro_batches_per_epoch} "
        f"optimizer_steps_per_epoch={optimizer_steps_per_epoch} "
        f"max_train_steps={max_train_steps} "
        f"warmup_steps={warmup_steps} "
        f"attn={attn_implementation} "
        f"model_dtype={model_dtype}"
    )

    global_step = 0
    completed_epochs = 0
    last_log_time = time.perf_counter()
    last_logged_step = 0

    for epoch in range(args.num_epochs):
        model.train()
        for batch in train_dataloader:
            with accelerator.accumulate(model):
                loss = compute_supervised_loss(
                    model,
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                    channelwise_loss_weight=channelwise_loss_weight,
                )
                accelerator.backward(loss)

                if accelerator.sync_gradients and args.max_grad_norm > 0:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                if accelerator.sync_gradients:
                    optimizer.step()
                    if not getattr(optimizer, "step_was_skipped", False):
                        lr_scheduler.step()
                    optimizer.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1
                if global_step % args.logging_steps == 0:
                    now = time.perf_counter()
                    steps_since_last_log = max(global_step - last_logged_step, 1)
                    elapsed = max(now - last_log_time, 1e-12)
                    last_log_time = now
                    last_logged_step = global_step
                    step_time = elapsed / steps_since_last_log
                    steps_per_sec = steps_since_last_log / elapsed
                    samples_per_sec = (global_batch_size * steps_since_last_log) / elapsed
                    eta_seconds = max(max_train_steps - global_step, 0) / steps_per_sec
                    logged_loss = accelerator.gather(loss.detach().float().reshape(1)).mean().item()
                    lr_val = lr_scheduler.get_last_lr()[0]
                    accelerator.print(
                        f"[{format_timestamp()}] "
                        f"epoch={epoch} step={global_step}/{max_train_steps} "
                        f"loss={logged_loss:.4f} "
                        f"lr={lr_val:.2e} "
                        f"step_time={step_time:.2f}s "
                        f"steps_per_sec={steps_per_sec:.3f} "
                        f"samples_per_sec={samples_per_sec:.2f} "
                        f"eta={format_duration(eta_seconds)}"
                    )

                if global_step >= max_train_steps:
                    break

        if (epoch + 1) % args.save_every_epochs == 0 or global_step >= max_train_steps:
            save_checkpoint(
                accelerator=accelerator,
                model=model,
                tokenizer=tokenizer,
                model_path=args.model_path,
                codec_path=args.codec_path,
                output_dir=output_root / f"checkpoint-epoch-{epoch + 1}",
                train_args=train_args_to_save,
                global_step=global_step,
                epoch=epoch + 1,
            )
        completed_epochs = epoch + 1

        if global_step >= max_train_steps:
            break

    save_checkpoint(
        accelerator=accelerator,
        model=model,
        tokenizer=tokenizer,
        model_path=args.model_path,
        codec_path=args.codec_path,
        output_dir=output_root / "checkpoint-last",
        train_args=train_args_to_save,
        global_step=global_step,
        epoch=completed_epochs,
    )
    accelerator.print(
        f"[{format_timestamp()}] [sft] finished "
        f"global_step={global_step} saved_epochs={completed_epochs} output_dir={output_root}"
    )


if __name__ == "__main__":
    main()
