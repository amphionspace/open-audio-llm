# 03c-sessions-v2：整场会议第二版关联

输入：03-sessions attempt 006 的窗口输出（不变），新增开发集 AISHELL-4 训练集前 10 场（只用于调关联参数）。

改动（`scripts/run_sessions.py`、`scripts/embed_worker_3dspeaker.py`、`configs/sessions-v2.yaml`）：

- 声纹：3D-Speaker ERes2NetV2（中文约 20 万说话人）替换 WeSpeaker ECAPA。离线诊断中 AISHELL-4 同人 / 异人 AUC 由 0.868 升到 0.910。
- 关联：两段式。先按 `high` 用平均链接聚出纯净簇；再把语音量不足整场 `small_fraction` 的小簇并进最相似、且不共享窗口的大簇。
- 选参：按模型在 AliMeeting Eval + AISHELL-4 开发集上取整场 cpER（微平均）最低的组合，测试集不参与。

结果（attempt 001 五个模型，002 补插值模型）：

| 模型 | 选中参数 | AliMeeting Test | AISHELL-4 Test | AMI（v2，英文） |
| --- | --- | --- | --- | --- |
| zh-emilia2-6000 | 0.40 / 0.10 | 23.7 → 22.6 | 40.8 → 34.5 | 39.2 → 43.8 |
| zh-dialogue-18000 | 0.40 / 0.03 | 25.0 → 24.3 | 44.0 → 35.5 | 39.6 → 43.0 |
| zh-farfield-14000 | 0.60 / 0.03 | 35.1 → 36.7 | 41.8 → 24.3 | 38.0 → 41.4 |
| zh-emilia2-6000-base10 | 0.70 / 0.03 | 29.4 | 29.2 | 45.7 |

说话人无关 CER 与第一版逐项相同（窗口输出复用正确）。中文声纹在英文 AMI 上更差，对外口径中 AMI 沿用第一版（ECAPA、单阈值）。插值 10% 版本在长会议窗口中复读 / 截断增多（AliMeeting Test 4 个复读窗口、8 个截断），说话人无关 CER 23.4 → 25.2。

W&B：benchmark-zh-20261011-sessions-v2-001 / -002。
