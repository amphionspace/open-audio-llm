# 训练旧数据对照组

输入：参见 [配置](../../configs/control-train.yaml) 中的显式路径和依赖。

完成条件：1000 步训练完成，有退出码、模型和 W&B 证据。

进展：历史执行已归档；执行状态与质量状态见各次记录。

运行：`open-audio-llm experiment run --config runs/clean-events-ab-20260928/experiment.yaml --task train-control`。

- [执行 001](attempts/001/README.md)
- [执行 002](attempts/002/README.md)
