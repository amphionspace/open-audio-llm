# 09 后续 2 评测与判定（回放 KL 加在场判断项）

- **输入**：任务 07、08 最后的 checkpoint，模型名 `kldreal`、`kldsynth`；11.17 已有的同口径 run（交付 27 组 `r1117`，纠偏 v2 think 关）。
- **做法**：`scripts/acp-node.sh deckl eval`，评测和比较的做法同任务 03、06；另和后续 1 同臂（kltreal、kltsynth）比较。
- **完成条件**：
  - 两臂的纠偏 v2 和交付 27 组全部退出码 0，W&B 远端都是 finished、有指标；
  - 按 README“后续 2”的判定和分支逐项给出结论。
- **进展**：待启动。
