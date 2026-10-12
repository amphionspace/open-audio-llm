# Emilia2 中文多人对话伪标签：ACP 单机 8 卡全参数

中文真实会议 cpER 连续三轮停在约 20–21（AliMeeting dev），真实会议数据只有几百小时、已反复使用；同口径下落后 MOSS 约 5–6 个点。本轮第一次加入大规模真实中文多人对话：`emilia2_zh_dialogue_diarize@moss-transcribe-diarize-0.9B`（本地 MOSS 0.9B 打说话人和文本，30–120 秒窗口按整窗保留：Qwen3-ASR 文字核对 CER ≤ 15%、说话人对照 Emilia2 自带短句）。

用户决定（2026-10-12）：单人 ASR 不是本项目看重的指标；本轮与单人 ASR 恢复轮（zh-asr-retention-acp-20261012）各占一台 8 卡机器并行。

## 配比

| 组 | 占比 | 内容 |
| --- | --- | --- |
| 对话轮全部来源 | 80 | 100 → 80，内部比例不变 |
| Emilia2 对话窗口 | 20 | 1–3 人窗口 192,658 个、约 5,026 h；不加噪声 |

4 人及以上的窗口（28,834 个、778 h）先排除（`shared/emilia2-exclude-4plus-speakers.txt`）：本地 0.9B 在这类窗口上只有 74%（4 人）和 51–60%（5 人以上）可用，MOSS Pro 重打版本 `moss-transcribe-diarize-0.9B-pro4plus` 完成后再补。窗口中 1 人的约占 19%。采样种子 48。

## 训练

与单人 ASR 恢复轮相同：1 节点 8 卡，每卡 2 条、梯度累积 2（全局 batch 32），学习率 5e-6、500 步 warmup 后 cosine 至 12000 步，每 1000 步保存；起点为对话轮选出的 18000 步，新优化器。

## 判定

开发机选模（GPU 2），每 1000 步，起点在同一组面板上实测作基线：

- 目标：AliMeeting dev 不计复读 cpER 最低。
- 门槛（都对起点）：6–12 人合成 cpER 不升 5 点以上；WenetSpeech dev、KeSpeech dev（普通格式，各 1000 条）各不升 1 点以上；复读不超过起点 + 6。
- 结束后对选中模型跑 meeting-180s、CHiME-6、固定 338、AISHELL-5 dev、CS-Dialogue test，并补测 benchmark-zh-20261011 三层（重点：整场会议 cpER）。

## 任务

- [01-prepare](tasks/01-prepare/)：数据配置。
- [02-train](tasks/02-train/)：ACP 训练。
- [03-select](tasks/03-select/)：开发机选模，停止文件 `tasks/03-select/control/stop-request.json`。
