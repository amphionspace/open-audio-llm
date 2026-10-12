# ACP 16 卡全参数训练（峰值学习率减半）

输入：8000 合并模型；上一轮冻结的数据配置；全新优化器、调度器和采样游标。

完成条件：16 ranks 实际训练、W&B 远端有训练指标；2000 步或收到停止请求时结束。

证据：`attempts/*/status.json`、`effective.yaml`、`artifacts/training/actual-training-contract.json`、`actual-learning-rates.json`。
