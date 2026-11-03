# AISHELL-5 素材复查

问题：排除有问题的会议、用齐来源会议后，AISHELL-5 素材是否可以和 AliMeeting 各占一半。

- 输入：[configs/synth-v3-fidelity-aishell5.yaml](../../configs/synth-v3-fidelity-aishell5.yaml)（AliMeeting 声学范围，因为对照的 dev 窗口是 AliMeeting）、`meeting_synth_fidelity@v6-20261101`（138 窗，放置时长是模板的 0.985）。
- 做法：起点模型经 vLLM 解码近讲、远场各 3 次，[configs/probe-aishell5.yaml](../../configs/probe-aishell5.yaml)。
- 证据：[attempts/001](attempts/001/artifacts/)；W&B `AmphionASR-zh-diar-synth-aishell5-material-001`，状态 `finished`，已核验 94 个指标。
- 完成条件：3 次解码完成，W&B 远端核验，和真实基线及 AliMeeting 素材对比。已完成。

## 结果

| 起点模型 | AISHELL-5 素材 | AliMeeting 素材 | 真实 |
| --- | --- | --- | --- |
| 近讲 cpER（含循环） | 61.2–70.1 | 18.9–19.8 | near_gated 16.35 |
| 近讲 cpER（不含循环） | 38.7–39.5 | 18.9–19.8 | |
| 远场 cpER（含循环） | 65.7–81.3 | 26.5–30.4 | far_mean 21.42；真实 AISHELL-5 Dev 45.7 |
| 远场 cpER（不含循环） | 41.1–43.3 | 25.9–26.5 | 真实 AISHELL-5 Dev 27.8 |
| 循环窗口（共 138 窗） | 近讲 6–8，远场 7–12 | 近讲 0，远场 0–1 | 真实 AISHELL-5 Dev 18/516 |
| 无重叠语句文字错误（近讲 / 远场） | 15.2 / 24.8 | 3.2 / 5.9 | |
| 无重叠语句说话人分裂（近讲） | +0.37 | +1.61 | |

真实 AISHELL-5 Dev 是车载远场，这里的时间轴和声学来自 AliMeeting，两者条件不同，只能看量级。

说话人数（多报 / 少报，%）：

| 窗口人数 | AISHELL-5 近讲 | AISHELL-5 远场 | AliMeeting 近讲 | AliMeeting 远场 |
| --- | --- | --- | --- | --- |
| 2 | 6.4 / 0 | 27.7 / 0 | 12.8 / 0 | 15.6 / 0 |
| 3 | 0 / 74.2 | 13.6 / 42.4 | 4.5 / 0 | 7.6 / 0 |
| 4 | 0 / 67.2 | 0.5 / 19.4 | 0 / 11.4 | 0.5 / 14.9 |

## 少报的原因排查

- 和拼接方式无关：4 人窗一场 4 人会议少报 75%，两场 2 人会议拼接少报 66.7%（[脚本](../../scripts/aishell5_merge_by_composition.py)）。
- 不是同一个人出现两次：全部窗口的人都不重复。
- 声音更相近，但解释不了少报（[脚本](../../scripts/aishell5_voice_similarity.py)，ERes2NetV2，每人最多 8 句取平均）：

| 窗口内最相近一对的相似度 | AISHELL-5 少报 | AliMeeting 少报 |
| --- | --- | --- |
| 0.3–0.4 | 66.7%（10 窗） | 5.9%（34 窗） |
| 0.4–0.5 | 56.9%（34 窗） | 4.6%（29 窗） |
| ≥ 0.5 | 79.4%（42 窗） | 18.5%（18 窗） |

平均成对相似度 AISHELL-5 0.37–0.41，AliMeeting 0.24–0.30。同样相似的声音，模型在 AISHELL-5 上也合并得多得多，所以主因不在声音分不开。具体机制未定，文字识别差（方言）可能是其中一部分。

## 决定

保持 AliMeeting 与 AISHELL-5 1:1，不再改配方。

- 没有发现标注缺陷：声纹能分开，人不重复，问题音轨已排除。起点模型做不好不等于标注错，这正是它需要学的。
- 训练只用远场（`dry_prob: 0`），少报比近讲低，3 人窗 42.4%，4 人窗 19.4%。
- 风险：这部分素材比真实数据难（不含循环 41–43，真实 AISHELL-5 Dev 27.8），占合成一半，可能占用模型能力而对 AliMeeting 类会议帮助不大。训练前测不出来，由选点规则和最终面板（含 `aishell5_dev`）判定。如果真实会议没有改善，先降 AISHELL-5 比例。
