# 03-liger-acp：ACP 上的端到端核对

- 输入：`configs/liger-acp-300.yaml`。由 ts-robust 030 的 `train-1node-v23.yaml` 派生：同数据配方（v7set6）、同起点（026 checkpoint-25000）、同种子、同学习率（前 300 步都在 500 步 warmup 内，与总步数无关），所以批次序列与 030 相同。代码为 030 的 `code-v10` 快照只换 `liger.py` / `train.py`（`shared/code-liger`）。唯一的训练差异是 `--use_liger_kernel true`；跑 300 步、不存 checkpoint。
- 完成条件：日志出现 `Liger kernels on the Qwen3-ASR text decoder: {...}`；300 步跑完；与 030 同步数对比 s/步、loss、显存峰值；W&B 远端核验通过。
- 进展：2026-10-19 07:56 UTC 提交，job `pt-srtumbsm`，排队中。
