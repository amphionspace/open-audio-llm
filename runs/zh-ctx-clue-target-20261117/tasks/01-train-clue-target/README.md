# 01-train-clue-target

- **输入**：`configs/train-clue-target.yaml`，数据 `configs/train-data-clue-target.yaml`（由 `scripts/make_configs.py` 生成）。think 文件换成 think-v3-20261117（线索 + 目标）。
- **完成条件**：3000 步完成，退出码 0；checkpoint-3000 带 `mtp/`；W&B 远端有 `train/loss`；日志里 MTP 参数组的学习率是 1e-4。
- **进展**：等 ACP 空位。
- **结果和证据**：执行记录在 `attempts/<编号>/`。
