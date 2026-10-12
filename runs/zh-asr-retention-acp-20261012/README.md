# 单人 ASR 恢复：ACP 单机 8 卡全参数

[全面评测](../benchmark-zh-20261011/) 显示，说话人转写链上的模型单人 ASR 比基座 Qwen3-ASR-1.7B 差 20–40%（WenetSpeech test_net 4.99 → 7.10，KeSpeech 5.10 → 6.70）。根因在数据：自统一格式起，训练里没有互联网口语类单人数据（WenetSpeech 因无说话人标注被排除），所有单人数据都改成 `[S1] 文本` 格式，普通转写格式不再训练；选模的保持指标只看 AliMeeting dev，没有覆盖单人 ASR。

格式核对（对话轮 18000 步，每集 500 条，同一服务两种请求）：

| 集合 | 普通格式 | 说话人格式 | 基座 |
| --- | --- | --- | --- |
| WenetSpeech test_net | 6.98 | 7.56 | 4.99 |
| KeSpeech test | 6.12 | 6.48 | 5.10 |
| CS-Dialogue 句级 test | 20.17 | 6.55 | 7.36 |

WenetSpeech、KeSpeech 两种格式都退化（识别能力本身）；CS-Dialogue 只在普通格式下退化（把英文说成中文）。因此新增数据用普通转写格式。

## 配比

| 组 | 占比 | 内容 |
| --- | --- | --- |
| 对话轮全部来源 | 75 | 100 → 75，内部比例不变 |
| WenetSpeech 清洗版 | 18 | `wenetspeech_clean@clean-v3-20260828`，9,971 h，`require_clean_pass` |
| 朗读/对话清洗版 | 7 | magicdata 4、aidatatang 1.5、primewords 1、aishell3 0.5（`clean-dual-qwen-moss-v1-20261008`，`require_clean_pass`） |

新增来源均为普通 `asr` 任务，0.5–20 秒。采样种子 47。

## 训练

用户要求只占一台（2026-10-12）：1 节点 8 卡，每卡 2 条、梯度累积 2，全局 batch 仍为 32；学习率 5e-6、500 步 warmup 后 cosine 至 12000 步（上一轮一半的样本量，时长相近）；每 1000 步保存。起点为对话轮选出的 18000 步，新优化器。

## 判定

开发机选模（GPU 6），每 1000 步评一次，起点在同一组面板上实测作基线：

- 目标：WenetSpeech dev 错误率（普通格式，固定 1000 条）最低。
- 门槛（都对起点）：AliMeeting dev 不计复读 cpER 不升 1 点以上；6–12 人合成 cpER 不升 5 点以上；KeSpeech dev 不升 0.5 点以上；复读不超过起点 + 6。
- 连续 3 个保存点都破同一门槛时停止。
- 结束后对选中模型跑 meeting-180s、CHiME-6、固定 338、AISHELL-5 dev、CS-Dialogue test，并补测 benchmark-zh-20261011 三层。

## 任务

- [01-prepare](tasks/01-prepare/)：数据配置。
- [02-train](tasks/02-train/)：ACP 训练。
- [03-select](tasks/03-select/)：开发机选模，停止文件 `tasks/03-select/control/stop-request.json`。
