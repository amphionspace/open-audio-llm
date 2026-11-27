# 05 后续训练臂 B（real-synth），CE 只训正文

- **输入**：同任务 02，只差 `ce_text_only: true`（README“后续：CE 只训正文”）；配置 `configs/train-real-synth-textce.yaml`。
- **完成条件**：
  - 训练跑完，`/workspace/data/ae-runs/ctx-kl-anchor-20261126/real-synth-textce/` 下有一半处和最后的 checkpoint；
  - W&B 远端 run 为 finished，并能看到 `train/ce_asr`、`train/replay_kl` 到最后一步。
- **进展**：
  - attempt 001（11-26 22:23–23:56Z，ACP pt-mweqkx4c）：完成，3,795 步，约 1.28 s/步，显存峰值 61 GiB。
    - checkpoint：`/workspace/data/ae-runs/ctx-kl-anchor-20261126/real-synth-textce/checkpoint-1898`（一半处）、`checkpoint-3795`（最后，评测用）。
    - CE（只算正文）：前 10% 0.371，最后 10% 0.056。回放 KL：中段 0.0181，最后 10% 0.0122（任务 02：0.0172 / 0.0126）。
    - 最后 10% 各类回放任务的 KL：TS 正样本 0.0179，`asr_no_kl` 0.0122，TS 负样本 0.0053，普通识别 0.0050，热词 0.0031。
    - W&B：<https://wandb.ai/1016097967-amphion/open-audio-llm/runs/AmphionASR-ctx-klanchor-textce-realsynth-from1117-ce-desc2075-synth2984x3-textonly-replaykl1-4gpu-001>，远端 finished，383 行到第 3795 步，最后一行（`train/ce_asr` 0.2293、`train/replay_kl` 0.0069）与本地一致。
- **结果 / 证据**：执行记录在 `attempts/<编号>/`（status.json、W&B 核验、`artifacts/training/logging.jsonl`、`replay-kl-by-task.jsonl`）。

<!-- execution-status -->

执行状态：completed；质量状态：未核实。

开始：2026-11-26T22:22:55.025752+00:00；结束：2026-11-26T23:56:20.305886+00:00；退出码：0。

<!-- /execution-status -->
