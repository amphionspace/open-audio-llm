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
  - attempt 002（17:35–19:00Z）：完成，3,795 步，约 1.0–1.4 s/步，显存峰值 61 GiB。
    - checkpoint：`/workspace/data/ae-runs/ctx-kl-anchor-20261126/real-synth/checkpoint-1898`（一半处）、`checkpoint-3795`（最后，评测用）。
    - CE：前 10% 0.338，最后 10% 0.050。回放 KL：前 10% 0.0035，中段 0.0159，最后 10% 0.0126。
    - 最后 10% 各类回放任务的 KL：TS 正样本 0.0194，`asr_no_kl` 0.0127，普通识别 0.0046，TS 负样本 0.0041，热词 0.0033。
    - W&B：<https://wandb.ai/1016097967-amphion/open-audio-llm/runs/AmphionASR-ctx-klanchor-realsynth-from1117-ce-desc2075-synth2984x3-replaykl1-4gpu-002>，远端 finished，383 行到第 3795 步，最后一行与本地一致。
- **结果 / 证据**：执行记录在 `attempts/<编号>/`（status.json、W&B 核验、`artifacts/training/logging.jsonl`、`replay-kl-by-task.jsonl`）。

<!-- execution-status -->

执行状态：completed；质量状态：未核实。

开始：2026-11-26T17:35:02.725343+00:00；结束：2026-11-26T19:00:01.758739+00:00；退出码：0。

<!-- /execution-status -->
