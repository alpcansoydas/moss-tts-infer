from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

import torch
from torch.utils.data import Dataset


USER_ROLE_PREFIX = "user\n"
USER_TEMPLATE_REFERENCE_PREFIX = "<user_inst>\n- Reference(s):\n"
USER_TEMPLATE_SUFFIX = "\n</user_inst>"
ASSISTANT_TURN_PREFIX = "\n"
ASSISTANT_ROLE_PREFIX = "assistant\n"

OPTIONAL_MESSAGE_FIELDS = (
    ("instruction", "Instruction"),
    ("tokens", "Tokens"),
    ("quality", "Quality"),
    ("sound_event", "Sound Event"),
    ("ambient_sound", "Ambient Sound"),
    ("language", "Language"),
)


def encode_text(tokenizer, text: str) -> List[int]:
    try:
        return list(tokenizer.encode(text, add_special_tokens=False))
    except TypeError:
        return list(tokenizer.encode(text))


def normalize_audio_codes(value: Any, field_name: str) -> torch.LongTensor:
    tensor = torch.as_tensor(value, dtype=torch.long)
    if tensor.ndim != 2:
        raise ValueError(f"`{field_name}` must have shape (T, n_vq), got {tuple(tensor.shape)}.")
    return tensor.cpu().contiguous()


def normalize_audio_code_list(
    value: Any,
    field_name: str,
    *,
    allow_none: bool = False,
) -> Optional[List[Optional[torch.LongTensor]]]:
    if value in (None, "", []):
        return None
    if torch.is_tensor(value):
        return [normalize_audio_codes(value, field_name)]
    if isinstance(value, list):
        if not value:
            return None
        if allow_none and any(item is None for item in value):
            return [
                None if item is None else normalize_audio_codes(item, f"{field_name}[{index}]")
                for index, item in enumerate(value)
            ]
        first_item = value[0]
        if torch.is_tensor(first_item):
            return [normalize_audio_codes(item, f"{field_name}[{index}]") for index, item in enumerate(value)]
        if isinstance(first_item, list):
            if first_item and isinstance(first_item[0], list):
                return [normalize_audio_codes(item, f"{field_name}[{index}]") for index, item in enumerate(value)]
            return [normalize_audio_codes(value, field_name)]
    raise TypeError(f"Unsupported `{field_name}` type: {type(value)}")


class MossTTSNanoSFTDataset(Dataset):
    def __init__(
        self,
        records: Iterable[Dict[str, Any]],
        *,
        tokenizer,
        model_config,
        max_length: int,
    ) -> None:
        self.records = list(records)
        self.tokenizer = tokenizer
        self.model_config = model_config
        self.max_length = int(max_length)
        if self.max_length < 8:
            raise ValueError("`max_length` must be >= 8.")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return self._build_example(self.records[index], index=index)

    def _build_example(self, record: Dict[str, Any], *, index: int) -> Dict[str, torch.Tensor]:
        if "text" not in record or not str(record["text"]).strip():
            raise ValueError(f"Record {index} is missing a non-empty `text` field.")
        if "audio_codes" not in record:
            raise ValueError(f"Record {index} is missing `audio_codes`. Run prepare_data.py first.")

        target_codes = self._normalize_codes_to_model_width(
            normalize_audio_codes(record["audio_codes"], "audio_codes"),
            field_name="audio_codes",
            index=index,
        )
        reference_codes = self._resolve_reference_codes(record, index=index)

        prompt_rows = self._build_prompt_rows(record=record, reference_codes=reference_codes)
        target_rows = self._build_audio_rows(
            target_codes,
            slot_token_id=self.model_config.audio_assistant_slot_token_id,
        )
        end_rows = self._build_text_rows([self.model_config.audio_end_token_id])
        full_sequence = torch.cat([prompt_rows, target_rows, end_rows], dim=0)
        prompt_length = int(prompt_rows.shape[0])

        if prompt_length >= self.max_length:
            raise ValueError(
                f"Record {index} prompt length {prompt_length} >= max_length {self.max_length}. "
                "Increase --max-length or shorten text/reference audio."
            )

        if full_sequence.shape[0] > self.max_length:
            full_sequence = full_sequence[: self.max_length]

        seq_len = int(full_sequence.shape[0])
        if seq_len < 2:
            raise ValueError(f"Record {index} packed sequence is too short: {seq_len}.")

        return {
            "full_input_ids": full_sequence,
            "seq_len": torch.tensor(seq_len, dtype=torch.long),
            "prompt_length": torch.tensor(prompt_length, dtype=torch.long),
        }

    def collate_fn(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        batch_size = len(batch)
        row_width = self.model_config.n_vq + 1

        full_input_ids = torch.full(
            (batch_size, self.max_length, row_width),
            int(self.model_config.audio_pad_token_id),
            dtype=torch.long,
        )
        full_input_ids[:, :, 0] = int(self.model_config.pad_token_id)
        full_attention_mask = torch.zeros((batch_size, self.max_length), dtype=torch.bool)
        loss_mask = torch.zeros((batch_size, self.max_length - 1), dtype=torch.bool)

        for batch_index, item in enumerate(batch):
            sequence = item["full_input_ids"]
            seq_len = int(item["seq_len"].item())
            prompt_length = int(item["prompt_length"].item())

            full_input_ids[batch_index, :seq_len, :] = sequence
            full_attention_mask[batch_index, :seq_len] = True
            loss_mask[batch_index, prompt_length - 1 : seq_len - 1] = True

        labels = full_input_ids[:, 1:, :].clone()
        labels = labels.masked_fill(~loss_mask.unsqueeze(-1), -100)
        labels = labels.masked_fill(~full_attention_mask[:, 1:].unsqueeze(-1), -100)
        labels[:, :, 1:] = labels[:, :, 1:].masked_fill(
            labels[:, :, 1:] == int(self.model_config.audio_pad_token_id),
            -100,
        )

        return {
            "input_ids": full_input_ids[:, :-1, :].contiguous(),
            "attention_mask": full_attention_mask[:, :-1].contiguous(),
            "labels": labels.contiguous(),
        }

    def _resolve_reference_codes(
        self,
        record: Dict[str, Any],
        *,
        index: int,
    ) -> Optional[List[Optional[torch.LongTensor]]]:
        if record.get("ref_audio_codes") is not None:
            codes_list = normalize_audio_code_list(record["ref_audio_codes"], "ref_audio_codes", allow_none=False)
            if codes_list is None:
                return None
            if len(codes_list) != 1:
                raise ValueError("MOSS-TTS-Nano finetuning only supports a single `ref_audio_codes` item per sample.")
            return [
                self._normalize_codes_to_model_width(codes_list[0], field_name="ref_audio_codes", index=index)
            ]

        if record.get("ref_audio") is not None:
            raise ValueError(
                f"Record {index} contains `ref_audio` but no precomputed `ref_audio_codes`. "
                "Run prepare_data.py first so training stays tokenizer-free."
            )
        return None

    def _build_prompt_rows(
        self,
        *,
        record: Dict[str, Any],
        reference_codes: Optional[List[Optional[torch.LongTensor]]],
    ) -> torch.LongTensor:
        prefix_ids = [self.model_config.im_start_token_id] + encode_text(
            self.tokenizer,
            USER_ROLE_PREFIX + USER_TEMPLATE_REFERENCE_PREFIX,
        )
        suffix_text = self._build_suffix_text(record)
        assistant_prefix_ids = encode_text(self.tokenizer, USER_TEMPLATE_SUFFIX + ASSISTANT_TURN_PREFIX) + [
            self.model_config.im_start_token_id
        ] + encode_text(self.tokenizer, ASSISTANT_ROLE_PREFIX)

        sections = [self._build_text_rows(prefix_ids)]
        if reference_codes is None:
            sections.append(self._build_text_rows(encode_text(self.tokenizer, "None" + suffix_text)))
        else:
            sections.append(self._build_text_rows([self.model_config.audio_start_token_id]))
            for reference in reference_codes:
                if reference is None:
                    sections.append(self._build_text_rows(encode_text(self.tokenizer, "None")))
                    continue
                sections.append(
                    self._build_audio_rows(reference, slot_token_id=self.model_config.audio_user_slot_token_id)
                )
            sections.append(
                self._build_text_rows([self.model_config.audio_end_token_id] + encode_text(self.tokenizer, suffix_text))
            )

        sections.append(self._build_text_rows(assistant_prefix_ids + [self.model_config.audio_start_token_id]))
        return torch.cat(sections, dim=0)

    def _build_suffix_text(self, record: Dict[str, Any]) -> str:
        lines = [""]
        for field_name, display_name in OPTIONAL_MESSAGE_FIELDS:
            value = record.get(field_name)
            lines.append(f"- {display_name}:")
            lines.append("None" if value in (None, "") else str(value))
        lines.append("- Text:")
        lines.append(str(record["text"]))
        return "\n".join(lines)

    def _build_text_rows(self, token_ids: List[int]) -> torch.LongTensor:
        rows = torch.full(
            (len(token_ids), self.model_config.n_vq + 1),
            int(self.model_config.audio_pad_token_id),
            dtype=torch.long,
        )
        if token_ids:
            rows[:, 0] = torch.tensor(token_ids, dtype=torch.long)
        return rows

    def _build_audio_rows(self, audio_codes: torch.LongTensor, *, slot_token_id: int) -> torch.LongTensor:
        rows = torch.full(
            (int(audio_codes.shape[0]), self.model_config.n_vq + 1),
            int(self.model_config.audio_pad_token_id),
            dtype=torch.long,
        )
        if rows.shape[0] > 0:
            rows[:, 0] = int(slot_token_id)
            rows[:, 1:] = audio_codes
        return rows

    def _normalize_codes_to_model_width(
        self,
        codes: torch.LongTensor,
        *,
        field_name: str,
        index: int,
    ) -> torch.LongTensor:
        target_width = int(self.model_config.n_vq)
        source_width = int(codes.shape[1])
        if source_width > target_width:
            raise ValueError(
                f"Record {index} field `{field_name}` has n_vq={source_width}, "
                f"but model expects at most {target_width}."
            )
        if source_width == target_width:
            return codes

        padded = torch.full(
            (int(codes.shape[0]), target_width),
            int(self.model_config.audio_pad_token_id),
            dtype=torch.long,
        )
        if source_width > 0:
            padded[:, :source_width] = codes
        return padded
