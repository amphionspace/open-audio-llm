# ACP 16 卡全参数训练

假设：上一轮复读与中文退化来自真实会话占比减半；恢复到 45% 并改用词级合成后，多人能力保留且复读不增加。主变量只有数据配比与合成版本。

输入：`01-prepare` 执行 002；8000 步合并模型；全新优化器、调度器与采样游标。

完成条件：16 ranks 实际训练、W&B 远端有训练指标；2000 步或收到停止请求时结束；checkpoint 上传。质量见实验 README。

进展：尚未提交。提交前完成 CPU 数据预检与 dry-run。

证据：`attempts/*/status.json`、`effective.yaml`、`wandb-start-verification.json`、`artifacts/training/actual-training-contract.json`、`actual-learning-rates.json`。

## 执行 001 失败与执行 002

- 执行 001（`pt-n16l54jx`）在第 486 步失败：worker 节点 rank 13 读取 `chime6/audio/train/S04_U01.wav` 时报“文件不存在”。同一时刻（19:50 PDT）CHiME-6 test 补标任务删除并重新生成了整个 CHiME-6 输出目录（含训练读取的 train 音频）。之后 master 节点的 rank 在集合通信中等待，因 `ddp_timeout` 为 14400 秒而约 2 小时未退出，直到平台重新调度。没有 checkpoint（首次保存在第 500 步），已停止该任务。
- 重新生成后的 CHiME-6 train 窗口数、时长和排除数与之前一致；执行 002 在任务内重新做数据预检。
- `ddp_timeout` 改为 1800 秒：评测已在开发机异步进行，训练不再需要长时间等待；保存与上传 checkpoint 远小于此值。其他训练参数与配比不变。
- 执行 002（`pt-y503ibhv`）从 8000 合并模型重新开始。训练期间不得改写训练正在读取的数据目录。
