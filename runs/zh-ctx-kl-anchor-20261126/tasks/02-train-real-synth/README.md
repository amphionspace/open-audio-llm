# 02 训练臂 B（real-synth）

- **输入**：
  - 起点 11.17（`/workspace/data/ae-state/model-cache/entries/model-afea989da22b77adc0ce9365`）；
  - 配置 `configs/train-real-synth.yaml`；
  - CE 流 `configs/ce-data-real-synth.yaml`：臂 A 的 2,075 条加合成说字数据 train 2,984 条，共 5,059 条；
  - 回放流 `configs/replay-data.yaml`：只算对 11.17 的 KL；
  - 代码 `runs/zh-ctx-grpo-20261120/shared/code-v3`（main 15eacbf）加 `scripts/ctx_kl_train.py`；同节点 GPU 4–7，臂 A 的数据预检做完以后再启动。
- **完成条件**：
  - 训练跑完，`/workspace/data/ae-runs/ctx-kl-anchor-20261126/real-synth/` 下有一半处和最后的 checkpoint；
  - W&B 远端 run 为 finished，并能看到 `train/ce_asr`、`train/replay_kl` 到最后一步。
- **进展**：
  - attempt 001（11-26 17:20–17:26Z）：启动在老师加载的修复之前（见 01 的 attempt 001），数据预检阶段用 SIGTERM 中断，没有训练。
  - attempt 002：臂 A 的 attempt 002 数据预检做完以后启动，用修复后的脚本。
- **结果 / 证据**：执行记录在 `attempts/<编号>/`（status.json、W&B 核验、`artifacts/training/logging.jsonl`、`replay-kl-by-task.jsonl`）。

<!-- execution-status -->

执行状态：interrupted；质量状态：未核实。

开始：2026-11-26T17:20:09.078280+00:00；结束：2026-11-26T17:26:51.580373+00:00；退出码：143。

<!-- /execution-status -->
