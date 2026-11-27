# 07 后续 2 训练臂 A（real），回放 KL 加在场判断项

- **输入**：同任务 04，只差 `decision_kl_weight: 1.0`（README“后续 2：回放 KL 加在场判断项”）；配置 `configs/train-real-deckl.yaml`。
- **完成条件**：
  - 训练跑完，`/workspace/data/ae-runs/ctx-kl-anchor-20261126/real-deckl/` 下有一半处和最后的 checkpoint；
  - W&B 远端 run 为 finished，并能看到 `train/ce_asr`、`train/replay_kl`、`train/replay_kl_decision` 到最后一步；
  - 生效核查（README“后续 2”）：第 1–3 步判断 token KL 只有数值差且不为 0；最后 10% TS 负样本判断 token KL 和任务 04 比。
- **进展**：
  - attempt 001（11-27 03:50Z 起，ACP pt-zmbgv4z9）：训练中。第 1 步 CE 0.17080、token 平均 KL 1.893e-4 与任务 04 相同，判断 token KL 1.9e-5（不为 0）。
- **结果 / 证据**：执行记录在 `attempts/<编号>/`。
