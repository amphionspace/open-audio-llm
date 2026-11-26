# 03 评测与判定

- **输入**：两臂最后的 checkpoint，模型名 `klreal`、`klsynth`；
  - 11.17 已有的同口径 run：交付 27 组 `r1117`，纠偏 v2 think 关。
- **做法**：`scripts/acp-node.sh eval`。
  - 纠偏 v2（think 关）：`runs/zh-ctx-grpo-k-20261122` 的 `make_eval_configs.py` 和 `run_eval.sh`，输出在 `/workspace/data/ae-runs/ctx-grpo-k-20261122-<名>`；
  - 交付 27 组：`runs/zh-ctx-grpo-20261120/scripts/acp-delivery-one.sh`，输出在 `/workspace/data/ae-runs/ctx-grpo-20261120-delivery-<名>`；
  - 比较：`compare_delivery.py r1117 <名>`，以及纠偏 v2 对 11.17 的配对比较和显式 / 隐式分组。
- **完成条件**：
  - 两臂的纠偏 v2 和交付 27 组全部退出码 0，W&B 远端都是 finished、有指标；
  - 按 README“判定”逐项给出结论。
- **进展**：待启动。
