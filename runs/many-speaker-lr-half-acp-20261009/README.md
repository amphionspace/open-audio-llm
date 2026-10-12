# 峰值学习率减半：ACP 16 卡全参数

只改一个变量：LLM/aligner 峰值学习率 1e-5 → 5e-6，音频编码器 5e-6 → 2.5e-6。数据配比、起点（8000 合并模型）、warmup 500 步 + cosine、全局 32 条、2000 步、每 500 步保存、2×8 A800 全部与 [many-speaker-real-mix-acp-20261009](../many-speaker-real-mix-acp-20261009/README.md) 相同；数据配置直接引用其 `01-prepare` 执行 002 的冻结产物，不重新组装。

## 为什么做

两轮全参训练中，中文都在峰值学习率附近退化最重、学习率衰减后回升：

| 轮次 | 中文最差点 | 后续 |
| --- | --- | --- |
| many-speaker-full-acp-20261008（旧配比） | 第 500 步，中文选模集比起点差 7.1 点 | 1500 步回到差 1.4 点 |
| many-speaker-real-mix-acp-20261009（新配比） | 第 1000 步，AliMeeting dev 37.49%（起点 22.12%） | 1500 步回到 25.72% |

新配比已使复读不再增加（17 → 9–11 条），6–12 人合成在第 500 步即由 112.51% 降到 42.86%。剩余的中文退化与学习率峰值同步出现，因此本轮只降峰值学习率，使结果可归因。梯度检查点的速度优化尚未实测显存，本轮不改。

## 判定与停止

沿用上一轮的新链路选模（vLLM 服务 + AmphionEval，独立面板）：AliMeeting dev 连续 2 次超过起点 + 1 点（23.12%）或复读条数连续 2 次超过起点 17 条即写停止文件；只在中文保持且复读不增的候选中选 6–12 人合成最低点。结束评测：meeting-180s、CHiME-6 dev、固定 338 条集，与起点和 MOSS 对照。执行完成与质量达标分开记录。

## 任务

- [02-train](tasks/02-train/README.md)：ACP 训练。
- [03-select](tasks/03-select/README.md)：开发机异步选模与结束评测，停止文件 `tasks/03-select/control/stop-request.json`。

正式入口：`open-audio-llm train --config runs/many-speaker-lr-half-acp-20261009/configs/train.yaml`。
