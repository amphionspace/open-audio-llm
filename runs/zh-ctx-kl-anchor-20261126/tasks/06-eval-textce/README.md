# 06 后续评测与判定（CE 只训正文）

- **输入**：任务 04、05 最后的 checkpoint，模型名 `kltreal`、`kltsynth`；11.17 已有的同口径 run（交付 27 组 `r1117`，纠偏 v2 think 关）。
- **做法**：`scripts/acp-node.sh textce eval`，评测和比较的做法同任务 03。
- **完成条件**：
  - 两臂的纠偏 v2 和交付 27 组全部退出码 0，W&B 远端都是 finished、有指标；
  - 按 README“后续：CE 只训正文”的判定逐项给出结论，并和 11-26 同臂比较。
- **进展**：待启动。
