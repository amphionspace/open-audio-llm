# ACP 全参数 24000 步（中文对话配比）

证据：`attempts/*/status.json`、`effective.yaml`、`artifacts/training/actual-training-contract.json`、`actual-learning-rates.json`。

执行 001（pt-omb3865v）在集群数据预检失败（采样窗口过小，一个来源配额为 0），未开始训练；执行 002（pt-6kaann1x）使用数据配置执行 003。
