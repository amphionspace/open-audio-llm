# Example configs

从仓库根目录运行。混合数据 YAML 使用分组注释，完整参数见 [数据配方说明](data/README.md)。
先将 MODEL 替换为本项目格式的
HF checkpoint，并确认 audio-data-contract 的 roots.json 指向本机实际数据。

| 文件 | 用途 |
|---|---|
| [data/catalog_smoke.yaml](data/catalog_smoke.yaml) | SFT / GRPO 共用的小规模在线数据 |
| [data/catalog_mixed.yaml](data/catalog_mixed.yaml) | SFT 多来源限量、重复、分片轮换、动态 batch、在线增强 |
| [train/sft.env](train/sft.env) | SFT 模型、GPU、优化器、冻结规则和恢复参数 |
| [train/grpo.env](train/grpo.env) | GRPO 生成分组、奖励训练和可选 vLLM 服务参数 |
| [train/rollout.env](train/rollout.env) | 独立 rollout server 参数 |

```bash
set -a
source examples/configs/train/sft.env
set +a
bash examples/train/sft/train.sh
```

先做快速检查时，在启动前覆盖以下变量：

```bash
export DATA_CONFIG=examples/configs/data/catalog_smoke.yaml
export MAX_STEPS=2 SAVE_STEPS=2 EVAL_STEPS=2 LOGGING_STEPS=1
bash examples/train/sft/train.sh
```

训练数据 YAML 与运行 env 分工不同：来源和增强写 YAML，设备、训练步数和优化器写 env。
SFT / GRPO 示例默认使用 GPU 2、3；独立 rollout 示例使用 GPU 3。并行启动 GRPO 与
rollout 时必须给它们分配不同 GPU，并调整训练进程数和生成 batch 大小。

原 configs/train 路径、离线 JSONL 训练脚本和 ShareGPT 生成器已移除。
当前入口和阶段冻结规则见 [训练复现](../../docs/train_reproduction.md)，
混合、增强及断点恢复的详细语义见 [在线训练数据](../../docs/online_training_data.md)。
