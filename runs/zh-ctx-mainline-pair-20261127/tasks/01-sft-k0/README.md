# 01 K0 SFT（K 的数据去掉纠偏部分）

- **输入**：`configs/train-k0.yaml`、`configs/train-data-k0.yaml`（`scripts/make_configs.py sft`），起点 11.17 交付包。
- **完成条件**：3000 步跑完，checkpoint-3000 上传到 `whai:open-audio-llm/runs/zh-ctx-mainline-pair-20261127/tasks/01-sft-k0/`；W&B 远端 finished、有指标。
- **进展**：
  - attempt 001（11-27 14:11–15:0xZ，ACP pt-f4eiiqi0）：数据预检失败，`Increase replay.window_samples to include every source`。
    - 原因：K 的窗口 2000 只是靠最大余数取整才通过。去掉三个来源以后，最小的来源（chime6 train_2spk）被取整成 0 个配额。kgrpo 的 GRPO 第一次也遇到过同样的问题。
    - 修正：按 kgrpo 的规则，窗口取 epoch_samples 的最小约数，并使每个来源的精确配额至少为 1，结果是 5000，最小配额 1.20。各来源占比不变，只改混合的粒度。
    - 预检的大部分时间（约 50 分钟）在重建 wenetspeech_clean 的索引（1,448 万条），重试时直接用缓存。
  - attempt 002（15:03–16:28Z，同一节点）：3000 步跑完，checkpoint-3000 已上传；W&B 远端 finished，global_step 3000，最后 train/loss 0.183（https://wandb.ai/1016097967-amphion/open-audio-llm/runs/AmphionASR-ctx-mainline-pair-k0-noctx-nothink-asrdialog-esreplay-final-mtp-from1117-3k-1node-codev1-002）。

<!-- execution-status -->

执行状态：completed；质量状态：未核实。

开始：2026-11-27T15:03:37.018613+00:00；结束：2026-11-27T16:28:24.451801+00:00；退出码：0。

对象存储：已同步到 `whai:open-audio-llm/runs/zh-ctx-mainline-pair-20261127/tasks/01-sft-k0/attempts/002`。

<!-- /execution-status -->
