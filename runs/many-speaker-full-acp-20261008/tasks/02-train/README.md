# ACP 16 卡全参数训练

假设：补齐 6–12 人、180 秒训练覆盖，可以改善人数与长程身份跟踪；英文只作少量回放。主变量为训练数据覆盖与配比。相比上一轮仍用 8000 合并起点及全部参数更新；新增有限训练期限与退化停止，避免无限续训。

输入：`01-prepare/004` 全部有效数据；8000 步合并模型；全新优化器、调度器和采样游标。

完成条件：ACP 两节点共 16 ranks 实际训练、W&B 远端有训练指标；每 500 步选模评测；2000 步或连续两次中文退化超过 1 点时结束；自动完成两套固定评测与权重上传。质量条件见实验 README，训练完成不自动代表质量达标。

进展：提交前完成 CPU 预检及启动配置核验。执行 001（`pt-qxzcopp7`）任务内数据预检通过，但构建 Trainer 时失败：ms-swift 以 `cls(args, trainer)` 构造回调，而 `ManySpeakerEvaluation.__init__` 不接收参数，16 个 rank 均未开始训练。修正构造签名并用桩对象核对 `on_train_begin` 后，执行 002（`pt-k08gzbgz`）重新提交，回调问题已解决，但两节点 rank 同时生成回放排列缓存时失败：quarkfs 上 flock 不能跨主机互斥，固定名 `.tmp` 文件被两边同时创建/改名。改为每个进程写独立临时文件后原子替换（排列确定，内容一致），既有 sampler 测试 23 passed，执行 003（`pt-731vpldk`）重新提交；旧执行与日志保留。状态与远端 W&B 证据以执行记录为准。

证据：`attempts/*/status.json`、`effective.yaml`、`wandb-start-verification.json`、`artifacts/training/actual-training-contract.json`、`actual-learning-rates.json`、`retention-objective.json`、`selection-history.json`、`final-results.json`。
