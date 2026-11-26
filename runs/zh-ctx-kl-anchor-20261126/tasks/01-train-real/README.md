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
  - attempt 002（17:27–18:09Z）：完成，1,557 步，约 1.35 s/步，显存峰值 59 GiB。
    - checkpoint：`/workspace/data/ae-runs/ctx-kl-anchor-20261126/real/checkpoint-778`（一半处）、`checkpoint-1557`（最后，评测用）。709 个张量里训练过的 700 个写回，冻结的 9 个保持 11.17 的值。
    - CE（每段日志的均值）：前 10% 0.301，最后 10% 0.095。回放 KL：第 1 步 1.9e-4（学生 fp32、老师 bf16 的数值差，说明老师是独立的模型），中段 0.0133，最后 10% 0.0130。
    - 最后 10% 各类回放任务的 KL：TS 正样本 0.0178，`asr_no_kl` 0.0134，TS 负样本 0.0059，普通识别 0.0032，热词 0.0031。
    - W&B：<https://wandb.ai/1016097967-amphion/open-audio-llm/runs/AmphionASR-ctx-klanchor-real-from1117-ce-desc2075x3-replaykl1-4gpu-002>，远端 finished，159 行到第 1557 步，最后一行与本地 `logging.jsonl` 一致。
- **结果 / 证据**：执行记录在 `attempts/<编号>/`（status.json、W&B 核验、`artifacts/training/logging.jsonl`、`replay-kl-by-task.jsonl`）。

<!-- execution-status -->

执行状态：completed；质量状态：未核实。

开始：2026-11-26T17:27:02.642871+00:00；结束：2026-11-26T18:09:30.663939+00:00；退出码：0。

<!-- /execution-status -->
