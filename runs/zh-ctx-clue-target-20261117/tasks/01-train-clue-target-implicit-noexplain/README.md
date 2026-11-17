# 01-train-clue-target-implicit-noexplain

- **输入**：`configs/train-clue-target-implicit-noexplain.yaml`、`configs/train-data-clue-target-implicit-noexplain.yaml`，隐式题数据构建完后由 `scripts/make_configs.py --implicit` 生成。数据是 clue-target 的数据加 `context_correction_synth_zh@implicit-v1-20261117`；隐式题的 think 开时只写固定句（`think-no-explanation.jsonl.gz`，由 `scripts/make_implicit_noexplain_think.py` 生成）。
- **完成条件**：3000 步完成，退出码 0；checkpoint-3000 带 `mtp/`；W&B 远端有 `train/loss`；日志里 MTP 参数组的学习率是 1e-4。
- **进展**：等隐式题数据和 ACP 空位。
- **结果和证据**：执行记录在 `attempts/<编号>/`。
