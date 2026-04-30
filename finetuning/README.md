# MOSS-TTS-Nano Finetuning Guide

This directory provides a complete finetuning workflow for `MOSS-TTS-Nano`:

- `prepare_data.py`: precomputes `audio_codes` for target audio and, when needed, `ref_audio_codes`
- `dataset.py`: packs fields such as `text / instruction / ambient_sound / ref_audio` into teacher-forcing samples
- `sft.py`: supports single-GPU, data parallel, and multi-node training
- `verify.py`: provides basic non-streaming inference checks
- `run_train.sh`: one-click wrapper that chains preprocessing and training

Default model weight locations:

- TTS model: `./models/MOSS-TTS-Nano`
- Audio codec: `./models/MOSS-Audio-Tokenizer-Nano`

## 1. Install Dependencies

From the repository root:

```bash
cd /path/to/MOSS-TTS-Nano
pip install -r requirements.txt
```

`requirements.txt` already includes:

- `accelerate>=1.0.0`
- `tqdm>=4.66.0`

## 2. Raw JSONL Format

The Nano finetuning pipeline mainly supports the following two formats.

### 2.1 Plain `text, speech` pairs

```jsonl
{"audio":"./data/utt0001.wav","text":"I realized that I am actually very good at noticing other people's emotions.","language":"en"}
{"audio":"./data/utt0002.wav","text":"She said she would be here by noon.","language":"en"}
```

### 2.2 Voice Cloning / Reference-Conditioned Training

Only one reference field is supported:

- `ref_audio`: a single reference audio clip

Example:

```jsonl
{"audio":"./data/utt0001.wav","text":"I realized that I am actually very good at noticing other people's emotions.","ref_audio":"./data/ref.wav","language":"en"}
{"audio":"./data/utt0002.wav","text":"She said she would be here by noon.","ref_audio":"./data/ref.wav","language":"en"}
```

### 2.3 Optional Fields

If needed, you can also provide the following fields. They will be appended to the user prompt:

- `instruction`
- `tokens`
- `quality`
- `sound_event`
- `ambient_sound`
- `language`

### 2.4 Path Rules

- Relative paths in the raw JSONL are resolved relative to the JSONL file location.
- Training expects preprocessed JSONL input, which means each record must already contain `audio_codes`.
- If reference-conditioned training is used, the training JSONL must also already contain `ref_audio_codes`.
- Nano finetuning currently supports only a single reference audio per sample.

## 3. Data Preprocessing

`prepare_data.py` does two things:

1. Encodes `audio` into `audio_codes`
2. Encodes `ref_audio` into `ref_audio_codes` by default

### 3.1 Single Process

```bash
python finetuning/prepare_data.py \
  --codec-path ./models/MOSS-Audio-Tokenizer-Nano \
  --input-jsonl train_raw.jsonl \
  --output-jsonl train_with_codes.jsonl \
  --batch-size 8
```

If you only want to encode target audio and skip reference audio:

```bash
python finetuning/prepare_data.py \
  --codec-path ./models/MOSS-Audio-Tokenizer-Nano \
  --input-jsonl train_raw.jsonl \
  --output-jsonl train_with_codes.jsonl \
  --skip-reference-audio-codes
```

### 3.2 Multi-Node / Multi-GPU Parallel Encoding

`prepare_data.py` follows the standard `accelerate launch` multi-process semantics.  
For example, with 2 nodes and 16 GPUs in total, the input is split into 16 shards and each rank writes its own output shard:

```bash
accelerate launch --num_processes 16 finetuning/prepare_data.py \
  --codec-path ./models/MOSS-Audio-Tokenizer-Nano \
  --input-jsonl train_raw.jsonl \
  --output-jsonl prepared/train_with_codes.jsonl
```

The outputs look like:

- `prepared/train_with_codes.rank00000-of-00016.jsonl`
- `prepared/train_with_codes.rank00001-of-00016.jsonl`
- ...
- `prepared/train_with_codes.rank00015-of-00016.jsonl`

During training, `sft.py` can directly read:

- a single JSONL file
- a directory
- a glob such as `prepared/train_with_codes.rank*.jsonl`
- or a comma-separated list of files

If your platform already injects multi-node communication environment variables, `accelerate launch` can usually reuse them directly.

## 4. Training

### 4.1 Single-GPU Baseline

```bash
accelerate launch finetuning/sft.py \
  --model-path ./models/MOSS-TTS-Nano \
  --codec-path ./models/MOSS-Audio-Tokenizer-Nano \
  --train-jsonl train_with_codes.jsonl \
  --output-dir output/moss_tts_nano_sft \
  --per-device-batch-size 1 \
  --gradient-accumulation-steps 8 \
  --learning-rate 1e-5 \
  --warmup-ratio 0.03 \
  --num-epochs 3 \
  --mixed-precision bf16 \
  --max-length 1024 \
  --channelwise-loss-weight 1,32
```

### 4.2 Single-Machine 8-GPU DDP

```bash
accelerate launch \
  --config_file finetuning/configs/accelerate_ddp_8gpu.yaml \
  finetuning/sft.py \
  --model-path ./models/MOSS-TTS-Nano \
  --codec-path ./models/MOSS-Audio-Tokenizer-Nano \
  --train-jsonl 'prepared/train_with_codes.rank*.jsonl' \
  --output-dir output/moss_tts_nano_sft_ddp \
  --per-device-batch-size 1 \
  --gradient-accumulation-steps 4 \
  --learning-rate 1e-5 \
  --num-epochs 3 \
  --mixed-precision bf16 \
  --max-length 1024 \
  --channelwise-loss-weight 1,32
```

### 4.3 Multi-Node Training

Update the following fields in your config file to match your cluster:

- `num_machines`
- `num_processes`
- `machine_rank`
- `main_process_ip`
- `main_process_port`

Keep the rest of the training command unchanged.

### 4.4 Important Arguments

- `--max-length`: fixed full sequence length. Samples are truncated to this length and then padded.
- `--channelwise-loss-weight`: supports two formats
  - `text_head,vq0,...,vqN`
  - `text_weight,total_audio_weight`
- `--save-every-epochs`: save one checkpoint every N epochs.

Single-GPU memory reference:

- With `accelerate launch --num_processes 1` and `--per-device-batch-size 1 --gradient-accumulation-steps 1 --max-length 1024 --mixed-precision bf16`, the measured training-process peak memory usage is about `3.23 GiB`.

### 4.5 Checkpoint Contents

Each checkpoint directory can be loaded directly by the inference code in this repository. It contains:

- model weights
- `config.json`
- tokenizer files
- the Nano model Python source files needed for loading
- `finetune_config.json`

## 5. One-Click Script

If you want a simple wrapper that chains preprocessing and training:

```bash
bash finetuning/run_train.sh
```

Common environment variables:

- `RAW_JSONL`: raw training JSONL
- `PREPARED_JSONL`: preprocessed JSONL
- `TRAIN_JSONL`: training input; if unset, it is inferred from `PREPARED_JSONL`
- `OUTPUT_DIR`: output directory
- `SKIP_PREPARE=1`: skip preprocessing and train directly
- `PREP_ACCELERATE_ARGS_STR`: extra `accelerate` args for `prepare_data.py`
- `TRAIN_ACCELERATE_ARGS_STR`: extra `accelerate launch` args for training, mainly for overriding `num_machines / num_processes / machine_rank`
- `PREP_EXTRA_ARGS_STR`: extra args passed to `prepare_data.py`
- `TRAIN_EXTRA_ARGS_STR`: extra args passed to `sft.py`
- `ACCELERATE_CONFIG_FILE`: training-time accelerate config file; if `TRAIN_ACCELERATE_ARGS_STR` is also provided, command-line values override the config defaults

Example:

```bash
RAW_JSONL=train_raw.jsonl \
PREPARED_JSONL=prepared/train_with_codes.jsonl \
OUTPUT_DIR=output/moss_tts_nano_sft \
PREP_ACCELERATE_ARGS_STR='--num_processes 8' \
ACCELERATE_CONFIG_FILE=finetuning/configs/accelerate_ddp_8gpu.yaml \
TRAIN_EXTRA_ARGS_STR='--per-device-batch-size 1 --gradient-accumulation-steps 4 --learning-rate 1e-5 --num-epochs 3 --mixed-precision bf16 --max-length 1024 --channelwise-loss-weight 1,32' \
bash finetuning/run_train.sh
```

For multi-node runs, the same idea applies: prepare shared encoded data first, then adjust `ACCELERATE_CONFIG_FILE` or `TRAIN_ACCELERATE_ARGS_STR` for your cluster.

## 6. Quick Verification

`verify.py` keeps the inference path intentionally simple. It supports:

- `voice_clone`: reference audio + target text
- `continuation`: continuation mode, with two input patterns
  - `prompt_text + prompt_audio_path + text`
  - or only `text`, which degrades to plain TTS

### 6.1 Voice Clone Verification

```bash
python finetuning/verify.py \
  --checkpoint output/moss_tts_nano_sft/checkpoint-last \
  --mode voice_clone \
  --text "This is a quick validation example for a finetuned model." \
  --prompt-audio-path ./assets/audio/zh_1.wav \
  --output-audio-path output/verify_voice_clone.wav
```

### 6.2 Continuation Verification

If `continuation` is used with `prompt-audio-path`, you must also provide the corresponding `prompt-text`:

```bash
python finetuning/verify.py \
  --checkpoint output/moss_tts_nano_sft/checkpoint-last \
  --mode continuation \
  --prompt-text "This sentence has already been spoken in the prompt audio." \
  --prompt-audio-path ./assets/audio/zh_1.wav \
  --text "This next sentence continues from that prompt for a quick continuation check." \
  --output-audio-path output/verify_continuation.wav
```

### 6.3 Plain TTS Verification

If you only want plain text-to-speech without reference audio, still use `continuation`, but do not pass `prompt-text` or `prompt-audio-path`:

```bash
python finetuning/verify.py \
  --checkpoint output/moss_tts_nano_sft/checkpoint-last \
  --mode continuation \
  --text "This is a quick non-streaming validation example." \
  --output-audio-path output/verify_tts.wav
```

You can also continue using the repository-level `infer.py`. Checkpoints saved by finetuning are already packaged in a format that `infer.py` can load directly.
