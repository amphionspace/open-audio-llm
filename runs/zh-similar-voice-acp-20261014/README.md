# 音色相近说话人：ACP 单机 8 卡全参数

计划见 [next-round-plan.md](../../docs/experiments/2026-10-14/next-round-plan.md)。AliMeeting dev（120 秒窗口，只看模型输出）上与 MOSS 的 5.0 点差距里，约 2.1 点是说话人归错；按窗口内最相近两人的声纹相似度分三档，归错差距为 0.9 / 2.2 / 3.9 点（选模器口径的高相似档：Emilia2 6000 步 4.02，MOSS 0.29）。重叠远场重排（zh-overlap-farsim-acp-20261013）没有改善重叠语句，本轮去掉。

## 状态

配置已写好，**未启动**。等两项上游：

1. amphiondata 注册音色相近视图 `similar-voice-v1-20261014`（选取结果：`/workspace/workspace/mingdong/data/amphiondata/similar-voice-views-20261014/`），注册后把 catalog 条目填进 `configs/prepare.yaml` 的 `catalogs`。
2. Emilia2 子集：声纹提取在跑；amphiondata 的并人检查出排除名单后剔除，再取前 1/3。

启动前再做 `shared/code` 代码快照（train.yaml、select.yaml 已指向本实验的快照路径）并选 GPU。

## 配比（采样占比 %）

| 组 | 重叠轮 | 本轮 |
| --- | --- | --- |
| 真实中文会议（含远场通道视图，30–600 秒） | 32.0 | 37 |
| 真实中文会议 · 音色相近子集 | 0 | 12 |
| Emilia2 对话 | 16.0 | 10 |
| Emilia2 对话 · 音色相近子集（未用过的部分） | 0 | 10 |
| Emilia2 重叠远场重排 | 20.0 | 0（移除） |
| 合成多人 | 15.1 | 12 |
| 单人 ASR 回放 | 10.5 | 12.6 |
| 近讲远场仿真 + 英文会议 | 6.4 | 6.4 |

音色相近子集内部按原真实会议各数据源的权重分配（AISHELL-4 12.8、AliMeeting 10.24、RAMC 3.2、AISHELL-5 3.2、CS-Dialogue 2.56），沿用原数据源的排除名单（427 个入选窗口在名单上）。

## 训练

起点 Emilia2 轮 6000 步。10% 原版权重插值（weight-interp-20261014）在选模面板上更好（AliMeeting dev 20.34），但在最终面板上变差（合成中文会议 10.2 → 18.1、AISHELL-5 dev 27.8 → 30.4、AliMeeting test 21.1 → 21.6），因此不作为起点，选中的 checkpoint 也不再做插值。

其余与前两轮相同：全局 batch 32，学习率 5e-6、500 步 warmup 后 cosine 至 12000 步，每 1000 步保存，新优化器。

## 判定

- 目标：AliMeeting dev 不计复读 cpER 最低。
- 门槛不变：6–12 人合成不升 5 点、WenetSpeech/KeSpeech dev 各不升 1 点、复读不超过起点 + 6。
- 验收（对起点，AliMeeting dev）：高相似 1/3 窗口的说话人归错（>3 字句子按字数汇总）≤ 2.0；cpER < 19.5；与他人重叠语句的字错不变差超过 0.5。
- 选模器加报：按说话人对齐后的字错 / 归错、高相似窗口归错、起点偏差 > 1 秒的语句占比；step 0 重新测，给这些新指标做基线。
