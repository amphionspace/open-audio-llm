# 对照轮：去掉音色相近加权（ACP 单机 8 卡全参数）

## 为什么做

[zh-similar-voice-acp-20261014](../zh-similar-voice-acp-20261014/README.md) 的 10 个保存点都没有超过起点，高相似窗口的说话人归错从 3.10 升到 3.5–5.8，原因没有定位（[总结](../../docs/experiments/2026-10-16/README.md)）。所有续训轮都在 20–21 附近，分不清两种可能：

1. 音色相近数据（占 22%）本身有害；
2. 从 Emilia2 6000 步用新优化器续训，本身就会退化。

本轮是对照：起点、步数、学习率、排除名单和其余配比都与上一轮相同，只去掉音色相近窗口的加权。

## 与上一轮的差别（采样占比 %）

| 组 | 上一轮 | 本轮 |
| --- | --- | --- |
| 真实中文会议 | 37 + 音色相近子集 12 | 49 |
| Emilia2 对话 | 10 + 音色相近子集 10 | 20 |
| 合成多人 / 单人 ASR / 近讲远场仿真 + 英文会议 | 12 / 12.6 / 6.4 | 不变 |

## 判定

- 与上一轮用同一个 step 0 基线（AliMeeting dev cpER 20.63，高相似归错 3.10），同一套选模和最终面板。
- 读法：本轮高相似归错也升到 5 以上 → 续训本身的问题，音色相近数据不是主因；本轮不升 → 音色相近数据有害，应换用法。差异小于 1 点不下结论。
- 本轮不以"超过起点"为验收：它是诊断，不是候选。

## 状态

2026-10-16：

- 数据配置 [01-prepare 002](tasks/01-prepare/attempts/002/)。001 的训练（pt-7w4xyjxv）在数据预检失败：注册表 `roots.json` 当天 20:13 把 `legacy_asr` 改指 `/workspace/data`，那里没有 `LHOTSE/data_aishell`（AISHELL-1），`prepare.py` 的 `moved` 逻辑覆盖了冻结配置里可用的本地化目录。002 用 `keep_roots: [legacy_asr]` 保留原路径。
- 训练 002 已提交 ACP 单机 8 卡（作业 `pt-iqa7526i`，[W&B](https://wandb.ai/1016097967-amphion/open-audio-llm/runs/zh-nosimilar-control-acp-20261016-002)）；选模 002 在开发机 GPU 2 上运行（[W&B](https://wandb.ai/1016097967-amphion/open-audio-llm/runs/zh-nosimilar-control-acp-20261016-select-002)）。选模 001 因训练 001 失败而失败。
- 作业显示名没有 `ASR-` 前缀（提交时规则尚未定下，作业提交后无法改名）。

## 结果

未完成。
