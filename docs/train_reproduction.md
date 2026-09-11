# 训练入口与复现

训练统一从 Catalog 在线构造数据。示例配置集中在
[examples/configs](../examples/configs/README.md)，每个参数附中文解释。

## SFT

从仓库根目录启动，先修改 env 中的 MODEL：

```bash
set -a
source examples/configs/train/sft.env
set +a
bash examples/train/sft/train.sh
```

快速检查时，在启动前覆盖：

```bash
export DATA_CONFIG=examples/configs/data/catalog_smoke.yaml
export MAX_STEPS=2 SAVE_STEPS=2 EVAL_STEPS=2 LOGGING_STEPS=1
bash examples/train/sft/train.sh
```

`DATA_CONFIG` 控制数据来源、混合与增强；env 控制设备、优化器、保存频率等训练参数。
SFT 混合能力、动态 batch 和断点恢复见 [在线训练数据](online_training_data.md)。
恢复时设置 `RESUME_FROM_CHECKPOINT=/path/to/checkpoint-N`，保持原配方与 batch 参数。

原有三阶段训练统一通过冻结参数选择，不再维护三套 JSONL 脚本：

| 训练阶段 | FREEZE_VIT | FREEZE_ALIGNER | FREEZE_LLM |
|---|---|---|---|
| encoder + connector | false | false | true |
| 仅语言模型 | true | true | false |
| 联合训练 | false | false | false |

阶段切换前先用 `examples/model/merge_lora.sh` 合并对应 adapter，将下一阶段 MODEL
指向合并模型。阶段间开始新训练；同一阶段中断才使用 RESUME_FROM_CHECKPOINT。

## GRPO

```bash
set -a
source examples/configs/train/grpo.env
set +a
bash examples/train/grpo/train.sh
```

示例默认本地生成；仍须将 MODEL 替换为本项目模型。GRPO 使用普通 Catalog
数据配方，不支持 SFT 专属 samples/reps、动态 batch 或 SpecAugment。
验证的全局 batch（`PER_DEVICE_EVAL_BATCH_SIZE × NPROC_PER_NODE`）须能被
`NUM_GENERATIONS` 整除；未设置每卡验证 batch 时，脚本默认使用 `NUM_GENERATIONS`。
可用 `PYTHON=/path/to/environment/bin/python` 选择环境；脚本优先加载当前 worktree
中的代码。额外命令行参数直接传给 ms-swift，包括
`--resume_from_checkpoint /path/to/checkpoint-N`。

如使用独立 rollout server，先配置 `examples/configs/train/rollout.env` 并启动
`examples/train/rollout/run_rollout_server.sh`，再设置 GRPO 的 USE_VLLM、服务地址和端口。
训练与 rollout 使用不同 GPU，调整 NPROC_PER_NODE 与可见卡数一致。

## 验证范围

数据链路测试覆盖音频按需读取、精确切段、在线增强、消息槽位顺序、混合比例、
分片轮换、DDP batch 计划及预取后的采样位置恢复。GPU 2、3 上已验证小规模 SFT
和中途恢复后 adapter 张量一致；这些检查不代表完整语料的 WER/CER 效果。

### GRPO 本地生成链路（2026-09-11）

已在 A800 80GB 上验证真实 Catalog 音频读取、分组生成、奖励计算、反向传播、
评估和 checkpoint 保存。环境为 Python 3.11.15、torch 2.10.0+cu128、
transformers 4.57.6、ms-swift 4.1.1、TRL 0.29.1、DeepSpeed 0.18.9、PEFT 0.18.1；
其他版本组合未作等价验证，不需要修改第三方库源码。

模型为项目格式的 Qwen3-ASR 音频编码器 + Qwen2.5-0.5B-Instruct，共约 684M 参数；
数据使用 `catalog_smoke.yaml` 的 LibriSpeech train/dev，保留在线变速增强。
在仓库根目录准备 `MODEL`、`PYTHON` 和 Catalog 路径后，可复现单卡检查：

```bash
CUDA_VISIBLE_DEVICES=0 NPROC_PER_NODE=1 MASTER_PORT=29541 \
DATA_CONFIG=examples/configs/data/catalog_smoke.yaml \
OUTPUT_DIR=runs/grpo-local-check MAX_STEPS=2 SAVE_STEPS=1 EVAL_STEPS=1 \
NUM_GENERATIONS=2 GENERATION_BATCH_SIZE=4 MAX_COMPLETION_LENGTH=96 \
LORA_RANK=8 LORA_ALPHA=16 DATALOADER_NUM_WORKERS=0 USE_VLLM=false \
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
bash examples/train/grpo/train.sh --report_to none
```

| 检查 | 结果 |
|---|---|
| 单卡、ZeRO-2、LoRA | 完成 2 步及每步评估、保存；loss 为 0.707788 / 0.710823，梯度范数为 3.770819 / 4.666790 |
| 权重更新 | checkpoint-1 到 checkpoint-2 有 168 个 adapter 张量变化，全部张量为有限值 |
| 双卡、每进程 2 个数据 worker | 完成 2 步、评估和保存，验证跨卡生成分组；配置同上，仅改可见 GPU、进程数和 worker 数 |
| 断点恢复 | 从单卡 checkpoint-1 恢复到第 3 步，完成后续训练、评估和保存；不承诺与不中断训练逐位一致 |

这验证的是默认 `USE_VLLM=false` 链路，不代表识别效果提升。独立 vLLM rollout
server、原生 Qwen3-ASR 热词/TS-ASR 配方未包含在此次验证中。
