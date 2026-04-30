# MOSS-TTS-Nano 微调教程

本目录提供基于 `MOSS-TTS-Nano` 架构的一套完整微调流程：

- `prepare_data.py`: 预提取训练目标音频的 `audio_codes`，并在需要时编码 `ref_audio_codes`
- `dataset.py`: 将 `text / instruction / ambient_sound / ref_audio` 等字段统一打包成 teacher-forcing 样本
- `sft.py`: 支持单卡、数据并行，以及多机训练
- `verify.py`: 做最基础的非流式快速推理验证
- `run_train.sh`: 串联预处理和训练的一键脚本

默认模型权重位置：

- TTS 模型: `./models/MOSS-TTS-Nano`
- 语音 codec: `./models/MOSS-Audio-Tokenizer-Nano`

## 1. 安装依赖

仓库根目录执行：

```bash
cd /path/to/MOSS-TTS-Nano
pip install -r requirements.txt
pip install "accelerate>=1.0.0" "tqdm>=4.66.0"
```

## 2. 原始 JSONL 格式

当前 Nano 版重点支持下面两种格式。

### 2.1 纯 `text, speech` pair

```jsonl
{"audio":"./data/utt0001.wav","text":"其实我真的有发现，我是一个特别善于观察别人情绪的人。","language":"zh"}
{"audio":"./data/utt0002.wav","text":"She said she would be here by noon.","language":"en"}
```

### 2.2 音色克隆 / 参考音频条件训练

只支持一个参考字段：

- `ref_audio`: 单条参考音频

单参考音频示例：

```jsonl
{"audio":"./data/utt0001.wav","text":"其实我真的有发现，我是一个特别善于观察别人情绪的人。","ref_audio":"./data/ref.wav","language":"zh"}
{"audio":"./data/utt0002.wav","text":"She said she would be here by noon.","ref_audio":"./data/ref.wav","language":"en"}
```

### 2.3 路径约定

- 原始 JSONL 里的相对路径，会自动按 JSONL 文件所在目录解析成绝对路径。
- 训练阶段要求输入是已经编码好的 JSONL，也就是记录里必须有 `audio_codes`。
- 如果用了参考音频训练，训练 JSONL 里也必须已经有 `ref_audio_codes`。
- 当前 Nano 微调只支持单参考音频，不支持多参考混合 prompt。

## 3. 预处理数据

`prepare_data.py` 会做两件事：

1. 把 `audio` 编成 `audio_codes`
2. 默认额外把 `ref_audio` 编成 `ref_audio_codes`

### 3.1 单进程

```bash
python finetuning/prepare_data.py \
  --codec-path ./models/MOSS-Audio-Tokenizer-Nano \
  --input-jsonl train_raw.jsonl \
  --output-jsonl train_with_codes.jsonl \
  --batch-size 8
```

如果只想编码目标音频，不编码参考音频：

```bash
python finetuning/prepare_data.py \
  --codec-path ./models/MOSS-Audio-Tokenizer-Nano \
  --input-jsonl train_raw.jsonl \
  --output-jsonl train_with_codes.jsonl \
  --skip-reference-audio-codes
```

### 3.2 多机多卡并行编码

`prepare_data.py` 现在直接按 `accelerate launch` 的多进程语义切分数据。  
例如 2 台节点、16 张卡，总共切 16 份，每个 rank 单独输出一个 shard：

```bash
accelerate launch --num_processes 16 finetuning/prepare_data.py \
  --codec-path ./models/MOSS-Audio-Tokenizer-Nano \
  --input-jsonl train_raw.jsonl \
  --output-jsonl prepared/train_with_codes.jsonl
```

输出会类似：

- `prepared/train_with_codes.rank00000-of-00016.jsonl`
- `prepared/train_with_codes.rank00001-of-00016.jsonl`
- ...
- `prepared/train_with_codes.rank00015-of-00016.jsonl`

后续训练阶段，`sft.py` 可以直接读取：

- 单个 JSONL
- 一个目录
- 一个 glob，例如 `prepared/train_with_codes.rank*.jsonl`
- 或逗号分隔的多个文件

如果你的平台已经自动注入了多机通信环境变量，`accelerate launch` 会直接复用这些信息；通常不再需要手动写 `torchrun` 风格的通信参数。

## 4. 启动训练

### 4.1 单卡基线

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

### 4.2 单机 8 卡 DDP

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

### 4.3 多机训练

将配置文件里的以下字段改成你的集群值即可：

- `num_machines`
- `num_processes`
- `machine_rank`
- `main_process_ip`
- `main_process_port`

例如 2 节点 16 卡，可以在两台机器分别设置：

- 节点 0: `machine_rank: 0`
- 节点 1: `machine_rank: 1`
- `num_machines: 2`
- `num_processes: 16`

其余训练命令保持不变。

### 4.4 重要参数

- `--max-length`: 固定 full sequence 长度。样本先按这个长度截断，再做 padding。
- `--channelwise-loss-weight`: 支持两种写法
  - `text_head,vq0,...,vqN`
  - `text_weight,total_audio_weight`
- `--save-every-epochs`: 每多少个 epoch 存一个 checkpoint。

单卡显存参考：

- 使用 `accelerate launch --num_processes 1`，并设置 `--per-device-batch-size 1 --gradient-accumulation-steps 1 --max-length 1024 --mixed-precision bf16` 做实测，训练进程峰值显存稳定，约为 `3.23 GiB`。


### 4.5 输出 checkpoint

每个 checkpoint 目录都可以直接被当前仓库的推理逻辑读取，里面会保存：

- 模型权重
- `config.json`
- tokenizer 文件
- Nano 模型所需的 Python 代码文件
- `finetune_config.json`

## 5. 一键脚本

如果你想用一套脚本把预处理和训练串起来，可以直接：

```bash
bash finetuning/run_train.sh
```

常用环境变量：

- `RAW_JSONL`: 原始训练集 JSONL
- `PREPARED_JSONL`: 预处理后的 JSONL
- `TRAIN_JSONL`: 训练输入；不填则自动从 `PREPARED_JSONL` 推断
- `OUTPUT_DIR`: 训练输出目录
- `SKIP_PREPARE=1`: 跳过预处理，直接训练
- `PREP_ACCELERATE_ARGS_STR`: 给 `prepare_data.py` 的 accelerate 参数
- `TRAIN_ACCELERATE_ARGS_STR`: 给训练阶段 `accelerate launch` 的参数，主要用于多机覆盖 `num_machines` / `num_processes` / `machine_rank`
- `PREP_EXTRA_ARGS_STR`: 给 `prepare_data.py` 的额外参数
- `TRAIN_EXTRA_ARGS_STR`: 给 `sft.py` 的额外参数
- `ACCELERATE_CONFIG_FILE`: 训练阶段的 accelerate 配置文件；如果同时传了 `TRAIN_ACCELERATE_ARGS_STR`，命令行参数会覆盖配置文件里的默认值

例如：

```bash
RAW_JSONL=train_raw.jsonl \
PREPARED_JSONL=prepared/train_with_codes.jsonl \
OUTPUT_DIR=output/moss_tts_nano_sft \
PREP_ACCELERATE_ARGS_STR='--num_processes 8' \
ACCELERATE_CONFIG_FILE=finetuning/configs/accelerate_ddp_8gpu.yaml \
TRAIN_EXTRA_ARGS_STR='--per-device-batch-size 1 --gradient-accumulation-steps 4 --learning-rate 1e-5 --num-epochs 3 --mixed-precision bf16 --max-length 1024 --channelwise-loss-weight 1,32' \
bash finetuning/run_train.sh
```

多机场景下也是同一套思路：先准备好共享可访问的编码数据，然后把 `ACCELERATE_CONFIG_FILE` 或 `TRAIN_ACCELERATE_ARGS_STR` 改成对应集群配置即可。

## 6. 快速推理验证

`verify.py` 只保留最基础的非流式验证，支持：

- `voice_clone`: 参考音频 + 目标文本
- `continuation`: 续写模式，支持两种输入
  - `prompt_text + prompt_audio_path + text`
  - 或者只给 `text`，退化为纯文本 TTS

### 6.1 voice clone 验证

```bash
python finetuning/verify.py \
  --checkpoint output/moss_tts_nano_sft/checkpoint-last \
  --mode voice_clone \
  --text "这是一个微调后模型的快速验证示例。" \
  --prompt-audio-path ./assets/audio/zh_1.wav \
  --output-audio-path output/verify_voice_clone.wav
```

### 6.2 continuation 续写验证

`continuation` 如果提供了 `prompt-audio-path`，就必须同时提供与该音频对应的 `prompt-text`：

```bash
python finetuning/verify.py \
  --checkpoint output/moss_tts_nano_sft/checkpoint-last \
  --mode continuation \
  --prompt-text "前面这一句是参考音频里已经说出来的内容。" \
  --prompt-audio-path ./assets/audio/zh_1.wav \
  --text "后面这一句继续往下生成，用来做续写验证。" \
  --output-audio-path output/verify_continuation.wav
```

### 6.3 纯 TTS 验证

如果你只想做不带参考音频的纯文本 TTS，也可以继续使用 `continuation` 模式，但这时不要传 `prompt-text` 和 `prompt-audio-path`：

```bash
python finetuning/verify.py \
  --checkpoint output/moss_tts_nano_sft/checkpoint-last \
  --mode continuation \
  --text "This is a quick non-streaming validation example." \
  --output-audio-path output/verify_tts.wav
```

如果你希望继续使用仓库根目录的 `infer.py` 也可以，训练输出的 checkpoint 已经按它可直接读取的结构保存好了。
