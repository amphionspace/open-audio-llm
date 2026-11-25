# 01 训练 noclean 臂

- **输入**：
  - 起点 11.06（`/workspace/data/ae-runs/amphionasr2-grpo-v10interp050/checkpoint`）；
  - 配置 `configs/train-noclean.yaml`，数据 `configs/train-data-noclean.yaml`（交付配方 v7set34 原版）；
  - 代码是 spelling-think 的 code-v1 快照（`oal-think-exp`，和对照组同一份）。
- **完成条件**：
  - 3000 步跑完，checkpoint 上传到 whai；
  - W&B 远端 run 为 finished，并能看到 `global_step` 3000。
- **进展**：2026-11-25 ACP `pt-p5ajf6pq` 在训练（attempt 001），15:30Z 打出第 1 步 loss。
  - W&B：https://wandb.ai/1016097967-amphion/open-audio-llm/runs/AmphionASR-ctx-cleanpair-noclean-from-v10interp050-v7set34-3k-1node-codev1-001
- **结果 / 证据**：执行记录在 `attempts/<编号>/`（status.json、W&B 核验）。

<!-- execution-status -->

执行状态：completed；质量状态：未核实。

开始：2026-11-25T14:33:08.385408+00:00；结束：2026-11-25T16:38:09.944441+00:00；退出码：0。

对象存储：已同步到 `whai:open-audio-llm/runs/zh-ctx-clean-pair-20261125/tasks/01-train-noclean/attempts/001`。

<!-- /execution-status -->
