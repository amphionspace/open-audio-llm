# 01 训练臂 A（real）

- **输入**：
  - 起点 11.17（`/workspace/data/ae-state/model-cache/entries/model-afea989da22b77adc0ce9365`）；
  - 配置 `configs/train-real.yaml`；
  - CE 流 `configs/ce-data-real.yaml`：5 个说明录音子集，2,075 条，终版 v2；
  - 回放流 `configs/replay-data.yaml`：只算对 11.17 的 KL；
  - 代码 `runs/zh-ctx-grpo-20261120/shared/code-v3`（main 15eacbf）加 `scripts/ctx_kl_train.py`；同节点 GPU 0–3。
- **完成条件**：
  - 训练跑完，`/workspace/data/ae-runs/ctx-kl-anchor-20261126/real/` 下有一半处和最后的 checkpoint；
  - W&B 远端 run 为 finished，并能看到 `train/ce_asr`、`train/replay_kl` 到最后一步。
- **进展**：
  - attempt 001（11-26 15:56–17:23Z，ACP pt-53size8p）：数据预检通过（回放 125 个来源新建索引约 84 分钟），第 1 步在老师前向上崩溃（`masked_scatter_size_check`），没有训练。
    - 原因：老师用 `copy.deepcopy(model)` 得到。`attach_sep_token` 把 thinker 原来的 `forward` 和 `get_audio_features` 存在闭包里，deepcopy 共用这些闭包，老师实际跑的是学生的 thinker，又拿不到 `enroll_n_frames`，TS 样本少插 sep token。即使不崩，“老师”也就是学生本身，KL 恒为 0。
    - 修复：老师改为单独再加载一份 11.17（`scripts/ctx_kl_train.py` `load()`）。
  - attempt 002：同一节点 17:27Z 起重跑，用修复后的脚本。
- **结果 / 证据**：执行记录在 `attempts/<编号>/`（status.json、W&B 核验、`artifacts/training/logging.jsonl`、`replay-kl-by-task.jsonl`）。

<!-- execution-status -->

执行状态：failed；质量状态：未核实。

开始：2026-11-26T15:56:37.048834+00:00；结束：2026-11-26T17:23:26.278649+00:00；退出码：1。

<!-- /execution-status -->
