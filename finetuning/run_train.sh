#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CONDA_ENV_PATH="${CONDA_ENV_PATH:-}"
MODEL_PATH="${MODEL_PATH:-${REPO_ROOT}/models/MOSS-TTS-Nano}"
CODEC_PATH="${CODEC_PATH:-${REPO_ROOT}/models/MOSS-Audio-Tokenizer-Nano}"

RAW_JSONL="${RAW_JSONL:-train_raw.jsonl}"
PREPARED_JSONL="${PREPARED_JSONL:-train_with_codes.jsonl}"
TRAIN_JSONL="${TRAIN_JSONL:-}"
OUTPUT_DIR="${OUTPUT_DIR:-output/moss_tts_nano_sft}"

PREP_DEVICE="${PREP_DEVICE:-auto}"
ACCELERATE_CONFIG_FILE="${ACCELERATE_CONFIG_FILE:-}"
SKIP_PREPARE="${SKIP_PREPARE:-0}"

PREP_ACCELERATE_ARGS_STR="${PREP_ACCELERATE_ARGS_STR:-}"
TRAIN_ACCELERATE_ARGS_STR="${TRAIN_ACCELERATE_ARGS_STR:-}"
PREP_EXTRA_ARGS_STR="${PREP_EXTRA_ARGS_STR:-}"
TRAIN_EXTRA_ARGS_STR="${TRAIN_EXTRA_ARGS_STR:---per-device-batch-size 1 --gradient-accumulation-steps 8 --learning-rate 1e-5 --num-epochs 3 --mixed-precision bf16 --max-length 1024 --channelwise-loss-weight 1,32}"

PREP_ACCELERATE_ARGS=()
TRAIN_ACCELERATE_ARGS=()
PREP_EXTRA_ARGS=()
TRAIN_EXTRA_ARGS=()

if [[ -n "${CONDA_ENV_PATH}" ]]; then
  PYTHON_BIN_DEFAULT="${CONDA_ENV_PATH}/bin/python"
  ACCELERATE_BIN_DEFAULT="${CONDA_ENV_PATH}/bin/accelerate"
  if [[ ! -x "${PYTHON_BIN_DEFAULT}" ]]; then
    echo "Missing python in CONDA_ENV_PATH: ${PYTHON_BIN_DEFAULT}" >&2
    exit 1
  fi
  if [[ ! -x "${ACCELERATE_BIN_DEFAULT}" ]]; then
    echo "Missing accelerate in CONDA_ENV_PATH: ${ACCELERATE_BIN_DEFAULT}" >&2
    exit 1
  fi
  export PATH="${CONDA_ENV_PATH}/bin:${PATH}"
fi

PYTHON_BIN="${PYTHON_BIN:-${PYTHON_BIN_DEFAULT:-python}}"
ACCELERATE_BIN="${ACCELERATE_BIN:-${ACCELERATE_BIN_DEFAULT:-accelerate}}"

if [[ -n "${PREP_ACCELERATE_ARGS_STR}" ]]; then
  read -r -a PREP_ACCELERATE_ARGS <<< "${PREP_ACCELERATE_ARGS_STR}"
fi
if [[ -n "${TRAIN_ACCELERATE_ARGS_STR}" ]]; then
  read -r -a TRAIN_ACCELERATE_ARGS <<< "${TRAIN_ACCELERATE_ARGS_STR}"
fi
if [[ -n "${PREP_EXTRA_ARGS_STR}" ]]; then
  read -r -a PREP_EXTRA_ARGS <<< "${PREP_EXTRA_ARGS_STR}"
fi
if [[ -n "${TRAIN_EXTRA_ARGS_STR}" ]]; then
  read -r -a TRAIN_EXTRA_ARGS <<< "${TRAIN_EXTRA_ARGS_STR}"
fi

derive_shard_glob() {
  local path="$1"
  local dir_name file_name stem suffix
  dir_name="$(dirname "${path}")"
  file_name="$(basename "${path}")"
  if [[ "${file_name}" == *.* ]]; then
    stem="${file_name%%.*}"
    suffix=".${file_name#*.}"
    printf '%s\n' "${dir_name}/${stem}.rank*${suffix}"
    return
  fi
  printf '%s\n' "${path}.rank*"
}

if [[ -z "${TRAIN_JSONL}" ]]; then
  TRAIN_JSONL="${PREPARED_JSONL}"
  if [[ -n "${PREP_ACCELERATE_ARGS_STR}" ]]; then
    TRAIN_JSONL="$(derive_shard_glob "${PREPARED_JSONL}")"
  elif [[ ! -e "${PREPARED_JSONL}" ]]; then
    SHARD_GLOB="$(derive_shard_glob "${PREPARED_JSONL}")"
    if compgen -G "${SHARD_GLOB}" > /dev/null; then
      TRAIN_JSONL="${SHARD_GLOB}"
    fi
  fi
fi

if [[ "${SKIP_PREPARE}" != "1" ]]; then
  if [[ -n "${PREP_ACCELERATE_ARGS_STR}" ]]; then
    "${ACCELERATE_BIN}" launch "${PREP_ACCELERATE_ARGS[@]}" "${REPO_ROOT}/finetuning/prepare_data.py" \
      --codec-path "${CODEC_PATH}" \
      --device "${PREP_DEVICE}" \
      --input-jsonl "${RAW_JSONL}" \
      --output-jsonl "${PREPARED_JSONL}" \
      "${PREP_EXTRA_ARGS[@]}"
  else
    "${PYTHON_BIN}" "${REPO_ROOT}/finetuning/prepare_data.py" \
      --codec-path "${CODEC_PATH}" \
      --device "${PREP_DEVICE}" \
      --input-jsonl "${RAW_JSONL}" \
      --output-jsonl "${PREPARED_JSONL}" \
      "${PREP_EXTRA_ARGS[@]}"
  fi
fi

if [[ -n "${ACCELERATE_CONFIG_FILE}" ]]; then
  "${ACCELERATE_BIN}" launch --config_file "${ACCELERATE_CONFIG_FILE}" "${TRAIN_ACCELERATE_ARGS[@]}" "${REPO_ROOT}/finetuning/sft.py" \
    --model-path "${MODEL_PATH}" \
    --codec-path "${CODEC_PATH}" \
    --train-jsonl "${TRAIN_JSONL}" \
    --output-dir "${OUTPUT_DIR}" \
    "${TRAIN_EXTRA_ARGS[@]}"
else
  "${ACCELERATE_BIN}" launch "${TRAIN_ACCELERATE_ARGS[@]}" "${REPO_ROOT}/finetuning/sft.py" \
    --model-path "${MODEL_PATH}" \
    --codec-path "${CODEC_PATH}" \
    --train-jsonl "${TRAIN_JSONL}" \
    --output-dir "${OUTPUT_DIR}" \
    "${TRAIN_EXTRA_ARGS[@]}"
fi
