# 08 后续 2 训练臂 B（real-synth），回放 KL 加在场判断项

- **输入**：同任务 05，只差 `decision_kl_weight: 1.0`（README“后续 2：回放 KL 加在场判断项”）；配置 `configs/train-real-synth-deckl.yaml`。
- **完成条件**：
  - 训练跑完，`/workspace/data/ae-runs/ctx-kl-anchor-20261126/real-synth-deckl/` 下有一半处和最后的 checkpoint；
  - W&B 远端 run 为 finished，并能看到 `train/ce_asr`、`train/replay_kl`、`train/replay_kl_decision` 到最后一步；
  - 生效核查（README“后续 2”）：第 1–3 步判断 token KL 只有数值差且不为 0；最后 10% TS 负样本判断 token KL 和任务 05 比。
- **进展**：待启动。
- **结果 / 证据**：执行记录在 `attempts/<编号>/`。
