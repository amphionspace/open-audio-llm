# 04 后续训练臂 A（real），CE 只训正文

- **输入**：同任务 01，只差 `ce_text_only: true`（README“后续：CE 只训正文”）；配置 `configs/train-real-textce.yaml`。
- **完成条件**：
  - 训练跑完，`/workspace/data/ae-runs/ctx-kl-anchor-20261126/real-textce/` 下有一半处和最后的 checkpoint；
  - W&B 远端 run 为 finished，并能看到 `train/ce_asr`、`train/replay_kl` 到最后一步。
- **进展**：
  - attempt 001（11-26 22:14–23:00Z，ACP pt-mweqkx4c）：完成，1,557 步，约 1.30 s/步，显存峰值 59 GiB。
    - checkpoint：`/workspace/data/ae-runs/ctx-kl-anchor-20261126/real-textce/checkpoint-778`（一半处）、`checkpoint-1557`（最后，评测用）。
    - 屏蔽生效：第 1 步与任务 01 的 attempt 002 是同一个 batch，CE 0.1285 → 0.1708（去掉的是容易预测的前缀 token）。
    - CE（每段日志的均值，只算正文）：前 10% 0.377，最后 10% 0.107。回放 KL：中段 0.0156，最后 10% 0.0135（任务 01：0.0152 / 0.0130）。
    - 最后 10% 各类回放任务的 KL：TS 正样本 0.0210，`asr_no_kl` 0.0138，TS 负样本 0.0052，普通识别 0.0033，热词 0.0027。
    - W&B：<https://wandb.ai/1016097967-amphion/open-audio-llm/runs/AmphionASR-ctx-klanchor-textce-real-from1117-ce-desc2075x3-textonly-replaykl1-4gpu-001>，远端 finished，159 行到第 1557 步，最后一行（`train/ce_asr` 0.0561、`train/replay_kl` 0.0227）与本地 `logging.jsonl` 一致。
- **结果 / 证据**：执行记录在 `attempts/<编号>/`（status.json、W&B 核验、`artifacts/training/logging.jsonl`、`replay-kl-by-task.jsonl`）。

<!-- execution-status -->

执行状态：completed；质量状态：未核实。

开始：2026-11-26T22:14:24.898334+00:00；结束：2026-11-26T23:00:47.418704+00:00；退出码：0。

<!-- /execution-status -->
