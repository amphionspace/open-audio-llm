# ACP 16 卡全参数训练

假设：上一轮复读与中文退化来自真实会话占比减半；恢复到 45% 并改用词级合成后，多人能力保留且复读不增加。主变量只有数据配比与合成版本。

输入：`01-prepare` 执行 002；8000 步合并模型；全新优化器、调度器与采样游标。

完成条件：16 ranks 实际训练、W&B 远端有训练指标；2000 步或收到停止请求时结束；checkpoint 上传。质量见实验 README。

进展：尚未提交。提交前完成 CPU 数据预检与 dry-run。

证据：`attempts/*/status.json`、`effective.yaml`、`wandb-start-verification.json`、`artifacts/training/actual-training-contract.json`、`actual-learning-rates.json`。
