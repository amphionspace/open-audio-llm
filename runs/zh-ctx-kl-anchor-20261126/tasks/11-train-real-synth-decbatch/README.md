# 11 后续 3 训练臂 B（real-synth），判断项用大批 TS 样本

- **输入**：同任务 05，只差判断项（README“后续 3：判断项用大批 TS 样本”）：每卡每步另取 16 条 TS 回放样本，回放损失 = token 平均 KL + 1.0 × 它们判断 token KL 的均值；配置 `configs/train-real-synth-decbatch.yaml`。
- **完成条件**：
  - 训练跑完，`/workspace/data/ae-runs/ctx-kl-anchor-20261126/real-synth-decbatch/` 下有一半处和最后的 checkpoint；
  - W&B 远端 run 为 finished，并能看到 `train/ce_asr`、`train/replay_kl` 到最后一步；`replay_kl_decision`、`decision_batch_kl` 训练后补录；
  - 生效核查（README“后续 3”）：第 1 步和任务 05 相同；第 300–500 步回放流 TS 负样本判断 token KL 不高于任务 05 同窗口的上限；最后 10% ≤ 0.010。
- **进展**：未开始。
- **结果 / 证据**：执行记录在 `attempts/<编号>/`（status.json、W&B 核验、`artifacts/training/logging.jsonl`、`replay-kl-by-task.jsonl`）。
