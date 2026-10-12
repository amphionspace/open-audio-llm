# 配置与启动

运行参数只写配置文件。模型、解释器、GPU、端口和任务参数在运行 YAML；来源版本、Catalog、roots 和增强在数据 YAML。路径用 `{path: ...}`，相对路径以配置文件所在目录为基准；模型 ID 等普通字符串保持原值。

| 配置 | 用途 |
|---|---|
| [train/sft.yaml](train/sft.yaml) | Catalog SFT |
| [train/sft-acp.yaml](train/sft-acp.yaml) | 同一 SFT 提交到 SenseCore ACP 算力池（2 节点 × 8 卡），见 [ACP 训练](../../docs/guides/acp.md) |
| [train/grpo.yaml](train/grpo.yaml) | 使用 vLLM rollout 的 GRPO |
| [train/qwen3-asr.yaml](train/qwen3-asr.yaml) | Qwen3-ASR 热词训练 |
| [train/ts-replay.yaml](train/ts-replay.yaml)、[ts-full.yaml](train/ts-full.yaml)、[ts-joint.yaml](train/ts-joint.yaml) | TS 回放与联合训练 |
| [train/sot.yaml](train/sot.yaml) | 全说话人转写 |
| [prepare/sot.yaml](prepare/sot.yaml)、[synthesize/sot.yaml](synthesize/sot.yaml) | 训练视图准备与合成 |
| [serve/vllm.yaml](serve/vllm.yaml)、[eval/comparison.yaml](eval/comparison.yaml) | vLLM 服务与评测 |
| [serve/tsasr.yaml](serve/tsasr.yaml) | 使用 SEP 和全局音频注意力的独立 TS-ASR 服务 |
| [rollout/vllm.yaml](rollout/vllm.yaml)、[model/](model/) | rollout 与模型处理 |
| [deploy/](deploy/) | 生成具体 Compose 部署文件 |

先填写模型和解释器路径，确认数据 YAML 的 `catalog`、`roots` 及 `metadata_cache`。启动或预览：

```bash
open-audio-llm train --config examples/configs/train/sft.yaml --dry-run
open-audio-llm train --config examples/configs/train/sft.yaml
open-audio-llm eval --config examples/configs/eval/comparison.yaml
open-audio-llm serve --config examples/configs/serve/vllm.yaml
```

CLI 只接受配置、任务选择、已有执行记录恢复与预览，不接受业务参数覆盖。shell 文件为薄入口，接受同样的 `--config`、`--dry-run`、`--resume`。旧 `MODEL`、`DATA_CONFIG`、`MAX_STEPS`、`AUDIO_DATA_CATALOG` 等环境变量不再生效。

`{attempt}` 在启动时替换为新执行目录。生效配置、数据配置快照、日志、退出码和 W&B 核验保存在该目录。训练和评测必须启用 W&B；凭据保留在环境中，由 CPU 记录器加载配置指定的 `~/.bashrc`，DDP 工作者不重复写状态。

GRPO 与 rollout 分配不同 GPU。TS/SOT 配方的 CPU 预检查使用与训练相同的数据配置、batch 和 world size。详见 [实验规范](../../docs/guides/experiments.md)。
