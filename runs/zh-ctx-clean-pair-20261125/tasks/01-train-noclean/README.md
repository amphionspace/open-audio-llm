# 01 训练 noclean 臂

- **输入**：
  - 起点 11.06（`/workspace/data/ae-runs/amphionasr2-grpo-v10interp050/checkpoint`）；
  - 配置 `configs/train-noclean.yaml`，数据 `configs/train-data-noclean.yaml`（交付配方 v7set34 原版）；
  - 代码是 spelling-think 的 code-v1 快照（`oal-think-exp`，和对照组同一份）。
- **完成条件**：
  - 3000 步跑完，checkpoint 上传到 whai；
  - W&B 远端 run 为 finished，并能看到 `global_step` 3000。
- **进展**：待提交。
- **结果 / 证据**：执行记录在 `attempts/<编号>/`（status.json、W&B 核验）。
