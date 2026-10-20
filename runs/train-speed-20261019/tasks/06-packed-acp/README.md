# 06-packed-acp：Liger + 打包的 ACP 端到端核对

- 输入：`configs/packed-acp-300.yaml`。由 ts-robust 029 的 `train-1node-v22.yaml` 派生：同数据配方（v7set4c）、同起点、同种子、同前 300 步学习率，批次序列相同；只加 `--use_liger_kernel` 与 `--packed_text`。代码 = 030 的 code-v10 快照 + 本分支 5 个文件（`shared/code-liger-packed`）。跑 300 步，不存 checkpoint。
- 完成条件：日志出现 Liger 替换数量和 "Packed text batches"；300 步跑完；与 029 前 300 步逐步对比 s/步、loss、显存峰值；W&B 远端核验通过。
- 进展：
  - attempt 001（`pt-ehku354l`，2026-10-19 10:34Z）：启动后 1 秒失败（exit 1），没有执行任何训练步。原因是配置的 `storage.local_root` 照抄了 029 的配置，仍指向 ts-robust 的 worktree，而执行目录在本 worktree 里。启动器在训练前的存储检查报错："attempt is outside storage.local_root"。`--dry-run` 不做这项检查。
  - 修正：`local_root` 改为本 worktree；在本地用 `storage.preflight_problems` 验证 attempt 002 路径，没有问题。
  - attempt 002（`pt-isk606t5`）：10:36Z 开始，300 步跑完；执行状态 completed，W&B 远端 finished（`wandb-verification.json`）。

## 结果（attempt 002 对比 029，`scripts/compare_runs.py`，证据见 `attempts/002/compare-029-*.json`）

| 项（第 101–300 步） | 029 | Liger + 打包 |
|---|---|---|
| 批次 | — | 每个日志区间的样本数和来源分布完全相同 |
| loss | — | 与 029 的相对差：平均 0.015%，最大 0.27% |
| token 填充 | 23.4% | 0% |
| 显存峰值 | 41.9 GB | 29.9 GB |
| 每个 microbatch 的 GPU 计算时间 | 0.540 s | 0.349 s（−35%） |
| 每个 microbatch 的主机侧时间 | 0.773 s | 0.741 s |
| 每步耗时 | 1.67 s | 1.71 s |

两边在第 300 步时都还在预热：029 的主机侧时间按 1.26 → 0.85 → 0.70 → 0.61 → 0.55 s 逐步下降，约第 1000 步后稳定在 1.1–1.2 s/步。

## 结论

- 数值正确：批次相同，loss 对齐，显存少 12 GB；GPU 计算少 35%。
- 每步耗时没有变短：瓶颈在主机侧（CPU 下发 kernel 的 Python 开销），GPU 省下的时间变成了等待，GPU 利用率从 92% 降到 64%。
- 按第 201–300 步的趋势（主机时间 0.666 s vs 0.698 s），稳态提速预计不超过约 5%。05 任务里"整步约 1.35–1.4×"的推算假设每步受 GPU 限制，这个假设不成立。
- 下一步要减少每个样本分摊的主机开销，例如每卡 batch 从 12 提到 24、梯度累积从 2 降到 1（全局 batch 不变，microbatch 数减半），利用省出来的显存和 GPU 余量。需要单独核对。
- 为什么对照 029：030 只跑到第 92 步就被挂起，没有 300 步的日志；029 与 030 起点、学习率、代码快照都相同，前 300 步日志齐全。
