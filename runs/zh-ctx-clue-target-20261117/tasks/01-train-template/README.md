# 01-train-template

- **输入**：`configs/train-template.yaml`，数据 `configs/train-data-template.yaml`（由 `scripts/make_configs.py` 生成）。think 文件与第二轮相同。
- **完成条件**：3000 步完成，退出码 0；checkpoint-3000 带 `mtp/`；W&B 远端有 `train/loss`；日志里 MTP 参数组的学习率是 1e-4。
- **进展**：2026-11-18 完成，3000 步、退出码 0，W&B finished，checkpoint 500–3000 带 `mtp/` 已上传 whai。MTP 学习率日志不按参数组记录，由生效参数 `joint_mtp_lr=0.0001` 和 MTP 准确率在骨干预热期就上升（第 70 步 0.80）间接核对。
- **结果和证据**：执行记录在 `attempts/<编号>/`。

<!-- execution-status -->

执行状态：completed；质量状态：未核实。

开始：2026-11-18T07:58:20.254394+00:00；结束：2026-11-18T09:24:24.289719+00:00；退出码：0。

对象存储：已同步到 `whai:open-audio-llm/runs/zh-ctx-clue-target-20261117/tasks/01-train-template/attempts/001`。

<!-- /execution-status -->
