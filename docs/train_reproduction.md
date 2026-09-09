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
如使用独立 rollout server，先配置 `examples/configs/train/rollout.env` 并启动
`examples/train/rollout/run_rollout_server.sh`，再设置 GRPO 的 USE_VLLM、服务地址和端口。
训练与 rollout 使用不同 GPU，调整 NPROC_PER_NODE 与可见卡数一致。

## 验证范围

数据链路测试覆盖音频按需读取、精确切段、在线增强、消息槽位顺序、混合比例、
分片轮换、DDP batch 计划及预取后的采样位置恢复。GPU 2、3 上已验证小规模 SFT
和中途恢复后 adapter 张量一致；这些检查不代表完整语料的 WER/CER 效果。
