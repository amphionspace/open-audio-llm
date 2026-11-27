# 04 评测与判定

- **输入**：k0grpo-avg3（03 第 200/300/400 步平均）、start-k0（01 的 checkpoint-3000）；对照 kgrpo-avg3、11.17 已有的 run。
- **做法**：`scripts/eval_after_training.sh`：纠偏 v2（think 关、开），交付 27 组（模型名 k0grpo）；比较用 `ae open-audio-llm compare` 和 `runs/zh-ctx-grpo-20261120/scripts/compare_delivery.py`。
- **完成条件**：全部退出码 0，W&B 远端 finished；按 README“判定”逐项给出结论。
- **进展**：未开始。
