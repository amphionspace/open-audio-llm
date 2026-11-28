# 08 第二个种子的评测与判定

- **为什么**：README“风险”第一条。K0 一轮的收益 CI 下限只有 +0.19，用第二个种子的 K0（k0s2grpo）估运行间波动。
- **输入**：k0s2grpo-avg3（07 第 200/300/400 步平均）、start-k0s2（05 的 checkpoint-3000）；对照 kgrpo、k0grpo、start-k0 已有的 run。
- **做法**：
  - 节点上 `scripts/eval_after_training.sh k0s2`（同 04：纠偏 v2 think 关 / 开，交付 27 组，模型名 k0s2grpo）。
  - 开发机上 `runs/zh-ctx-grpo-20261120/scripts/compare_delivery.py k0s2grpo kgrpo` 和 `compare_delivery.py k0grpo k0s2grpo`，然后 `scripts/gate_seed2.py`。它用 gate.py 同一套规则算三对，结果写到本目录：
    - `k0s2grpo-vs-kgrpo/`：判定再算一遍；
    - `k0grpo-vs-k0s2grpo/`：两边都不带纠偏，只看波动；
    - `start-k0-vs-start-k0s2/`：两个 K0 SFT 的纠偏 v2，只记录。
- **完成条件**：节点脚本退出码 0，评测 run 在 W&B 远端 finished；三对结果写出，结论写进本 README 和实验 README。
- **进展**：等 07 的 GRPO 和评测跑完（预计 11-28 04:00–05:00Z）。gate.py 加了 `OUT` / `TITLE`，列名改用 `MODELS`；默认输出和已提交的 `tasks/04-eval/gate.md`、`gate.json` 逐字节相同（开发机上重跑核对过）。
