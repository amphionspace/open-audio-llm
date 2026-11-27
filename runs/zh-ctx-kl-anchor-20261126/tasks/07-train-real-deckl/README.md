# 07 后续 2 训练臂 A（real），回放 KL 加在场判断项

- **输入**：同任务 04，只差 `decision_kl_weight: 1.0`（README“后续 2：回放 KL 加在场判断项”）；配置 `configs/train-real-deckl.yaml`。
- **完成条件**：
  - 训练跑完，`/workspace/data/ae-runs/ctx-kl-anchor-20261126/real-deckl/` 下有一半处和最后的 checkpoint；
  - W&B 远端 run 为 finished，并能看到 `train/ce_asr`、`train/replay_kl`、`train/replay_kl_decision` 到最后一步；
  - 生效核查（README“后续 2”）：第 1–3 步判断 token KL 只有数值差且不为 0；最后 10% TS 负样本判断 token KL 和任务 04 比。
- **进展**：
  - attempt 001（11-27 03:50–04:37Z，ACP pt-zmbgv4z9）：完成，1,557 步，约 1.48 s/步，显存峰值 59 GiB。
    - checkpoint：`/workspace/data/ae-runs/ctx-kl-anchor-20261126/real-deckl/checkpoint-778`（一半处）、`checkpoint-1557`（最后，评测用）。
    - 生效核查第 1 条通过：第 1 步与任务 04 是同一个 batch，CE 0.17080、token 平均 KL 1.893e-4 相同；判断 token KL 第 1–3 步为 1.9e-5、≈0、6.6e-4（不为 0）。整个训练中“每个回放样本第 1 个监督 token 是 `language`、每个样本恰有一个判断 token”的断言都没触发。
    - **生效核查第 2 条没过：判断 token KL 不降反升。**
      - 两次训练同种子，每个记录窗口的回放样本相同（159 个窗口的任务组成全部一致），所以可以逐窗口配对。在这些相同的回放样本上（每个样本第一次出现时、更新前测，相当于留出估计）：
        - TS 负样本 token 平均 KL：任务 04 最后 10% 为 0.0050，本任务 0.0117；
        - 判断 token KL：任务 04 没记，上限是 4 × 0.0050 = 0.020（负样本只有 4 个监督 token）；本任务为 0.0467，几乎全部 KL 都在判断 token 上。
        - 后半程 79 个窗口里有 73 个是本任务更高。
      - 两边在第 100–250 步分开，也就是 warmup 升到峰值学习率那一段（第 100 步还相同，第 250 步 TS 负样本 0.0018 vs 0.0313）。
    - CE（每段日志的均值，只算正文）：前 10% 0.371，最后 10% 0.111（任务 04：0.371 / 0.105）。
    - 回放 token 平均 KL：中段 0.0185，最后 10% 0.0156（任务 04：0.0150 / 0.0137）。
    - 判断 token KL（全部回放任务）：中段 0.0378，最后 10% 0.0238。
    - 最后 10% 各类回放任务：

      | 任务 | token 平均 KL（括号为任务 04） | 判断 token KL |
      |---|---|---|
      | TS 正样本 | 0.0216（0.0216） | 0.0219 |
      | TS 负样本 | 0.0117（0.0050） | 0.0467 |
      | `asr_no_kl` | 0.0154（0.0143） | 0.0187 |
      | 普通识别 | 0.0041（0.0031） | 0.0006 |
      | 热词 | 0.0029（0.0027） | 0.0002 |

    - 梯度范数（裁剪前，阈值 1.0，每步都被裁剪）：中位数 17.1，最后 10% 12.2（任务 04：12.4 / 5.6）。
    - 相对 11.17 的权重改动在每组模块上都比任务 04 小约 5%：音频塔 0.353% vs 0.360%；LLM 第 0–6、7–13、14–20、21–27 层分别是 0.345 / 0.363 / 0.253 / 0.087% vs 0.365 / 0.387 / 0.269 / 0.092%。所以判断项改的是更新方向，不是更新幅度（`decision_rows_delta.py`，日志 `/workspace/workspace/mingdong/scratch/ctx-kl-anchor/decision-rows-delta.log`）。
    - W&B：<https://wandb.ai/1016097967-amphion/open-audio-llm/runs/AmphionASR-ctx-klanchor-deckl-real-from1117-ce-desc2075x3-textonly-replaykl1-decisionkl1-4gpu-001>。
      - 远端 finished，159 行到第 1557 步，最后一行 `train/ce_asr` 0.0608、`train/replay_kl` 0.0270，与本地一致；
      - `train/replay_kl_decision` 不在启动器同步的白名单里，训练结束后（11-27 05:50Z）从本地 `logging.jsonl` 补录到同一个 run（按 `global_step`）；远端 159 行，第 1 步 1.9e-5、第 1557 步 0.0454，与本地一致。
- **结果 / 证据**：执行记录在 `attempts/<编号>/`（status.json、W&B 核验、`artifacts/training/logging.jsonl`、`replay-kl-by-task.jsonl`）。

<!-- execution-status -->

执行状态：completed；质量状态：未核实。

开始：2026-11-27T03:49:57.885885+00:00；结束：2026-11-27T04:36:37.916033+00:00；退出码：0。

<!-- /execution-status -->
