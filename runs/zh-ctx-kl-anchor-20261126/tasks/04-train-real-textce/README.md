# 04 后续训练臂 A（real），CE 只训正文

- **输入**：同任务 01，只差 `ce_text_only: true`（README“后续：CE 只训正文”）；配置 `configs/train-real-textce.yaml`。
- **完成条件**：
  - 训练跑完，`/workspace/data/ae-runs/ctx-kl-anchor-20261126/real-textce/` 下有一半处和最后的 checkpoint；
  - W&B 远端 run 为 finished，并能看到 `train/ce_asr`、`train/replay_kl` 到最后一步。
- **进展**：待启动。
- **结果 / 证据**：执行记录在 `attempts/<编号>/`。
