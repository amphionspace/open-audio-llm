# 01-train-clue-target-implicit

- **输入**：`configs/train-clue-target-implicit.yaml`、`configs/train-data-clue-target-implicit.yaml`，隐式题数据构建完后由 `scripts/make_configs.py --implicit` 生成。数据是 clue-target 的数据加 `context_correction_synth_zh@implicit-v1-20261117`；隐式题的 think 写语义线索 + 目标（`think.jsonl.gz`）。
- **完成条件**：3000 步完成，退出码 0；checkpoint-3000 带 `mtp/`；W&B 远端有 `train/loss`；日志里 MTP 参数组的学习率是 1e-4。
- **进展**：等隐式题数据和 ACP 空位。
- **结果和证据**：执行记录在 `attempts/<编号>/`。

<!-- execution-status -->

执行状态：failed；质量状态：未核实。

开始：2026-11-18T11:12:54.887161+00:00；结束：2026-11-18T11:15:14.379629+00:00；退出码：1。

对象存储：已同步到 `whai:open-audio-llm/runs/zh-ctx-clue-target-20261117/tasks/01-train-clue-target-implicit/attempts/001`。

<!-- /execution-status -->
