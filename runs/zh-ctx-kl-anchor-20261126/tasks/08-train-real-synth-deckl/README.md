# 08 后续 2 训练臂 B（real-synth），回放 KL 加在场判断项

- **输入**：同任务 05，只差 `decision_kl_weight: 1.0`（README“后续 2：回放 KL 加在场判断项”）；配置 `configs/train-real-synth-deckl.yaml`。
- **完成条件**：
  - 训练跑完，`/workspace/data/ae-runs/ctx-kl-anchor-20261126/real-synth-deckl/` 下有一半处和最后的 checkpoint；
  - W&B 远端 run 为 finished，并能看到 `train/ce_asr`、`train/replay_kl`、`train/replay_kl_decision` 到最后一步；
  - 生效核查（README“后续 2”）：第 1–3 步判断 token KL 只有数值差且不为 0；最后 10% TS 负样本判断 token KL 和任务 05 比。
- **进展**：
  - attempt 001（11-27 03:58–05:33Z，ACP pt-zmbgv4z9）：完成，3,795 步，约 1.12 s/步，显存峰值 61 GiB。
    - checkpoint：`/workspace/data/ae-runs/ctx-kl-anchor-20261126/real-synth-deckl/checkpoint-1898`（一半处）、`checkpoint-3795`（最后，评测用）。
    - 生效核查第 1 条通过：第 1 步与任务 05 是同一个 batch，CE 0.5564、token 平均 KL 1.9e-4 相同；判断 token KL 第 1–3 步为 1.9e-5、≈0、7.9e-4（不为 0）。断言都没触发。
    - **生效核查第 2 条没过：判断 token KL 不降反升。**
      - 与任务 05 逐窗口配对（`scripts/replay_kl_paired.py`），383 个窗口的回放样本相同。
        - TS 负样本 token 平均 KL：任务 05 最后 10% 为 0.0054，本任务 0.0137；
        - 判断 token KL：任务 05 的上限是 4 × 0.0054 = 0.022，本任务为 0.0546。
        - 后半程 192 个窗口里有 167 个、最后 10% 的 38 个里有 37 个是本任务更高。
      - 两边在第 100–480 步之间分开，第 480 步 TS 负样本 0.0035 vs 0.1128。
    - CE（只算正文）：前 10% 0.381，最后 10% 0.058（任务 05：0.379 / 0.056）。
    - 回放 token 平均 KL：中段 0.0217，最后 10% 0.0151（任务 05：0.0178 / 0.0121）。
    - 判断 token KL（全部回放任务）：中段 0.0349，最后 10% 0.0221。
    - 最后 10% 各类回放任务：

      | 任务 | token 平均 KL（括号为任务 05） | 判断 token KL |
      |---|---|---|
      | TS 正样本 | 0.0254（0.0179） | 0.0215 |
      | TS 负样本 | 0.0137（0.0054） | 0.0546 |
      | `asr_no_kl` | 0.0153（0.0121） | 0.0152 |
      | 普通识别 | 0.0063（0.0050） | 0.0034 |
      | 热词 | 0.0040（0.0031） | 0.0021 |

    - 梯度范数（裁剪前，阈值 1.0）：中位数 12.8，最后 10% 9.8（任务 05：7.9 / 4.2）。
    - 相对 11.17 的权重改动在每组模块上都比任务 05 小约 5%：音频塔 0.534% vs 0.547%；LLM 四段 0.514 / 0.539 / 0.382 / 0.132% vs 0.551 / 0.580 / 0.406 / 0.138%。
    - W&B：<https://wandb.ai/1016097967-amphion/open-audio-llm/runs/AmphionASR-ctx-klanchor-deckl-realsynth-from1117-ce-desc2075-synth2984x3-textonly-replaykl1-decisionkl1-4gpu-001>。
      - 远端 finished，383 行到第 3795 步，最后一行 `train/ce_asr` 0.2379、`train/replay_kl` 0.0084，与本地一致；
      - `train/replay_kl_decision` 训练结束后从本地日志补录：远端 383 行，第 3795 步 0.0040，与本地一致。
- **结果 / 证据**：执行记录在 `attempts/<编号>/`（status.json、W&B 核验、`artifacts/training/logging.jsonl`、`replay-kl-by-task.jsonl`）。

<!-- execution-status -->

执行状态：completed；质量状态：未核实。

开始：2026-11-27T03:58:27.922087+00:00；结束：2026-11-27T05:32:50.666365+00:00；退出码：0。

<!-- /execution-status -->
