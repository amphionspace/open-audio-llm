# ACP 2 节点冒烟训练（2026-10-08）

目的：首次在 SenseCore ACP 算力池（`cluster-whai`）上，用 2 节点 × 8 张 A800 跑 Qwen3-ASR-1.7B 的 30 步 LoRA SFT，验证多机 torchrun、W&B 远端记录、checkpoint 上传 `whai:open-audio-llm` 和执行记录同步。不评估模型能力。

| 任务 | 说明 |
| --- | --- |
| [01-train](tasks/01-train/README.md) | aishell@legacy-20260804 训练 30 步，每 10 步保存，提交到 ACP 排队 |

数据：Catalog 副本 `configs/catalog/aishell.yaml` 与 2026-10-05 对象存储冒烟实验相同；aishell 没有 clean 训练版本，使用非 clean 训练集。用法见 [ACP 训练](../../docs/acp.md)。

<!-- execution-status -->

执行状态：partial；质量状态：未核实。

开始：未知；结束：未结束；退出码：未知。

<!-- /execution-status -->
