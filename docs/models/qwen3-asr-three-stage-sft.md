# Qwen3-ASR-1.7B 三阶段 SFT

从 `Qwen3-ASR-1.7B` 做三轮全参 SFT，得到 `checkpoint-34479`。三轮都是 1 个 epoch、bf16、DeepSpeed ZeRO-1、4 卡。没有使用 LoRA。评测见 [checkpoint-34479 评测结果](ckpt34479-evaluation.md)。

| 阶段 | 运行目录 | 起点 | 结束 checkpoint | 步数 | 学习率 | eval_loss |
|---|---|---|---|---:|---:|---:|
| 1 | `sft/v1-20260914-071419` | Qwen3-ASR-1.7B | checkpoint-79731 | 79731 | 2e-5 | 0.11220 |
| 2 | `sft/v2-20260921-223309` | 阶段 1 的 checkpoint-79731 | checkpoint-33407 | 33407 | 1e-5 | 0.19186 |
| 3 | `sft/continue_sft_turn_taking_police/v0-20260924-205747` | 阶段 2 的 checkpoint-33407 | checkpoint-34479 | 34479 | 1e-5 | 0.10534 |

阶段 1 在同一次输出目录里从 checkpoint-62000 恢复，最终仍是从底座训完的 1 个 epoch。阶段 2 按 `eval_loss` 的 best 是 checkpoint-33000（0.19182），阶段 3 使用的是 epoch 结束时的 checkpoint-33407（0.19186）。阶段 3 的 best 与最终步都是 checkpoint-34479，对应 eval token accuracy 0.97038。

## 三阶段共用

| 项目 | 配置 |
|---|---|
| 模型类型 | `qwen3_asr`，`tuner_type=full`，`torch_dtype=bfloat16` |
| 优化器 | AdamW，β=(0.9, 0.95)，weight decay 0.1，grad clip 1.0 |
| 调度 | cosine，`min_lr=1e-6`，warmup 3000 步，`warmup_ratio=0` |
| 长度 | `max_length=2048`，`truncation_strategy=delete`，不做 packing |
| batch | 每卡 14，梯度累积 2，4 卡，有效 batch 112；验证每卡 8 |
| 并行 | DeepSpeed ZeRO-1，不 offload optimizer，`sequence_parallel_size=1` |
| 训练 | 1 epoch，`max_steps=-1`，gradient checkpointing，lazy tokenize |
| 数据 | shuffle，seed / data_seed 42，`split_dataset_ratio=0.01` |
| 保存 | 每 1000 步保存并验证，`save_total_limit=5`，`logging_steps=10` |
| 冻结 | `thinker.model.embed_tokens`，以及 audio tower 的 `conv2d1`、`conv2d2`、`conv2d3`、`conv_out` |
| 额外可训练 | `thinker.audio_tower.proj1`、`proj2`、`sep_token` |
| 模块开关 | `freeze_vit=false`，`freeze_llm=false`，`freeze_aligner=false` |

encoder、aligner 和 LLM 其余参数随全参更新。`target_modules=all-linear` 与 `lora_rank=64` 留在 args 里，但 `tuner_type=full`，没有训练 LoRA。

名字带 `eval_` 的 jsonl 写在训练 `dataset` 列表里，不是单独的 `val_dataset`。Trainer 再从合并后的数据划出 1% 做验证。文件名后缀 `#N` 表示该文件最多取 N 条。

## 阶段 1：普通 ASR、热词、TS-ASR 与回放

起点：`Qwen3-ASR-1.7B`。学习率 2e-5。13 个数据文件，全部全量读取。

- 热词：`hw_sft/hw_sft.jsonl`
- TS-ASR：LibriMix 960 的 `train_2spk`、`train_3spk`、`train_neg`；AISHELLMix pack3 的 `train_2spk`、`train_3spk`、`train_neg`
- 回放与普通 ASR：`replay_v3_3000h_qwen3asr.jsonl`、`voice-in-the-wild-2m/vitw.jsonl`
- 会议与其他 ASR：AISHELL-4、MagicData RAMC、AliMeeting、GigaSpeech-M 的 `train.jsonl`

第一步验证 eval_loss 为 0.37290，结束时为 0.11220。

## 阶段 2：话轮转换，并下采样阶段 1 数据

起点：checkpoint-79731。学习率降到 1e-5。在阶段 1 的数据上增加话轮转换，并用 `#N` 限制旧数据条数。

全量读取的话轮数据：

- AliMeeting：train/eval 的 2spk、3spk、neg
- AISHELL-4：train 的 2spk、3spk、neg
- AMI SDM：train/eval 的 2spk、3spk、neg
- RAMC：train/eval 的 2spk、neg（没有 3spk）
- CHiME-6：train/eval 的 2spk、3spk、neg
- Emilia2 dialog：train/eval 的 2spk、3spk、neg

下采样后的阶段 1 数据：

| 文件 | 上限 |
|---|---:|
| `hw_sft/hw_sft.jsonl` | 104330 |
| LibriMix `train_2spk` / `train_3spk` / `train_neg` | 54384 / 81222 / 8202 |
| AISHELLMix `train_2spk` / `train_3spk` / `train_neg` | 65220 / 97830 / 9783 |
| `replay_v3_3000h_qwen3asr.jsonl` | 294019 |
| `voice-in-the-wild-2m/vitw.jsonl` | 55642 |
| AISHELL-4 / MagicData RAMC / AliMeeting `train.jsonl` | 9700 / 15420 / 15237 |
| GigaSpeech-M `train.jsonl` | 91014 |

结束 eval_loss 0.19186。按 eval_loss 更低的 checkpoint-33000 没有作为下一阶段起点。

## 阶段 3：继续话轮，并加入警务与短语音

起点：checkpoint-33407。学习率仍为 1e-5。话轮文件除 Emilia2 dialog 外继续全量读取；Emilia2 dialog 改为上限抽样。阶段 1 的回放上限有的提高、有的保持。同一警务文件在列表里重复出现，按重复次数加权。

Emilia2 dialog 上限：train 2spk / 3spk / neg 为 45652 / 17978 / 35456，eval 2spk / 3spk / neg 为 898 / 400 / 726。

相对阶段 2 调整的回放上限：

| 文件 | 阶段 2 | 阶段 3 |
|---|---:|---:|
| `hw_sft/hw_sft.jsonl` | 104330 | 695545 |
| LibriMix `train_2spk` / `train_3spk` / `train_neg` | 54384 / 81222 / 8202 | 127346 / 190188 / 19206 |
| AISHELLMix `train_2spk` / `train_3spk` / `train_neg` | 65220 / 97830 / 9783 | 152719 / 229078 / 22908 |
| `replay_v3_3000h_qwen3asr.jsonl` | 294019 | 294019 |
| `voice-in-the-wild-2m/vitw.jsonl` | 55642 | 106015 |
| MagicData RAMC `train.jsonl` | 15420 | 15420 |
| AISHELL-4、AliMeeting `train.jsonl` | 有上限 | 全量 |

阶段 3 新增、且全量读取：

- `ami_sdm/train.jsonl`
- `police_v6_8h/train.jsonl`
- `police_incremental_20260916/train.jsonl`

阶段 3 新增、且列表中重复 3 次：

- `police_v6_expanded/train.jsonl`
- `police_terms_v5/train.jsonl`
- `police_synthetic_zh_accent/train.jsonl`

另外各取最多 461812 / 461126 条：`emilia2-short/zh.jsonl`、`emilia2-short/en.jsonl`。GigaSpeech-M 不再进入这一阶段。

结束 eval_loss 0.10534，eval token accuracy 0.97038。这一步就是评测使用的 checkpoint-34479。
