# 中文对话配比：ACP 16 卡全参数

[中文远场配比](../zh-farfield-mix-acp-20261010/README.md) 从 4000 到 18000 步的中文真实会议 cpER 一直在 20.1–21.6 之间（平台期），在 18000 步选模后停止，从其选模最优的 14000 步（多人目标 30.53，中文 20.32）继续。用户决定（2026-10-10）：中文优先、英文远场不关键、能用的数据都用；WenetSpeech 由用户上传，不等待。

| 组 | 占比 | 变化 |
| --- | --- | --- |
| 上一轮保留部分 | 84 | AISHELL-4、AliMeeting、RAMC、词级合成、全部回放，比例不变 |
| 英文会议 | 3 | 6 → 3（AMI、NOTSOFAR、CHiME-6 内部比例不变） |
| 旧 1-5 人合成 | 4 | 10 → 4 |
| AISHELL-5 | 5 | 新增：车载 2-4 人，4 个座舱远场麦克风各作一个视角 |
| CS-Dialogue | 4 | 新增：两人中英混说电话对话（近场），按说话人静音合成单声道混合 |

采样种子 46；其余训练参数与上一轮相同：学习率 5e-6、500 步 warmup 后 cosine 至 24000 步，新优化器，2×8 A800，每 2000 步保存并在开发机 GPU 6 选模。

## 为什么做

连续三轮只调配比/学习率，真实会议指标都在约 6000 步后平台。策略复盘结论：关键杠杆是更大规模的真实中文多人对话，而不是继续微调配比。本轮先用已有的公开中文对话（AISHELL-5、CS-Dialogue）扩充真实数据；后续 WenetSpeech + 付费 API 伪标注作为下一步。

## 数据要点

- 构建代码：amphiondata 分支 `feat/zh-dialogue-corpora`（`aishell5.py`、`cs_dialogue.py`、`meetings.py` 预设 `aishell5`/`cs_dialogue`）。
- 已注册到 `/workspace/data/registry`（`zh_dialogue.yaml`），同步 audio-data-contract PR #28。
- AISHELL-5 train 包从 OpenSLR trmal/openslr.org 分段下载；elda 镜像同尺寸内容不同，已弃用，损坏分片按 zlib 流定位后重取。
- 两组新数据均排除空目标窗口；都无 clean 版本，使用现有训练集。

## 判定

与之前相同：只在排除复读的 AliMeeting dev cpER 连续 2 次高于基线（8000 合并模型，22.10%）+ 1 点时停止；复读条数不超过 17 + 6 的 checkpoint 才作为候选。结束评测 meeting-180s（重点看中文真实会议）、CHiME-6 dev、固定 338 条集，以及新增的 AISHELL-5 dev 与 CS-Dialogue test（检验新数据本身的收益，起点 14000 步结果见 04-start-eval），与起点和 MOSS 对照。执行完成与质量达标分开记录。

## 任务

- [01-prepare](tasks/01-prepare/README.md)：数据配置。
- [02-train](tasks/02-train/README.md)：ACP 训练。
- [03-select](tasks/03-select/README.md)：开发机选模，停止文件 `tasks/03-select/control/stop-request.json`。
- [04-start-eval](tasks/04-start-eval/README.md)：起点在新增结束评测集上的结果。
