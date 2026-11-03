# 用齐来源会议的所有人（whole_conversations）的复查

- 改动：`OnlineMeetingConfig.whole_conversations`（`src/open_audio_llm/data/meeting_online.py`）。窗口用齐来源会议的所有人：2 人窗口用 2 人会议，3 人窗口用 3 人会议，4 人窗口用 4 人会议或两场 2 人会议；凑不出（1 人窗口）时用最小的会议。只有一个声音的会议（对方音轨未过检查）不算完整。
- 输入：`meeting_synth_fidelity@v5-20261101`（[configs/synth-v2-fidelity-alimeeting.yaml](../../configs/synth-v2-fidelity-alimeeting.yaml)，只用 AliMeeting 素材）、起点模型、[configs/probe-whole.yaml](../../configs/probe-whole.yaml)。
- 完成条件：近讲、远场各解码 3 次并合并，W&B 远端可查（`AmphionASR-zh-diar-synth-whole-conversations-001`，已 finished）。
- 执行：attempt 001 完成，exit 0。

## 结果（138 窗，3 次解码合并）

| 指标 | v1 | 用齐 | 真实对照 |
| --- | --- | --- | --- |
| 近讲 cpER | 22.4–23.4 | **18.9–19.8** | near_gated 16.35 |
| 远场 cpER（不计循环） | 30.4–30.9 | **25.9–26.5** | far_mean 21.42 |
| 说话人数正确率 近讲 / 远场 | 0.68 / 0.66 | **0.89 / 0.86** | 0.91 / 0.92 |
| 多报 2 人窗口 近讲 / 远场 | 61.7% / 62.4% | **12.8% / 15.6%** | |
| 多报 3 人窗口 近讲 / 远场 | 31.8% / 50.0% | **4.5% / 7.6%** | |
| 无重叠归属额外 近讲 / 远场 | +5.63 / +6.06 | **+3.39 / +3.54** | +1.02 / +1.98 |
| 无重叠文字 近讲 / 远场 | 6.15 / 10.15 | **3.24 / 5.92** | 4.45 / 6.16 |
| 重度重叠文字 近讲 / 远场 | 30.9 / 44.1 | 30.3 / 43.3 | 25.1 / 35.0 |

- 远场 r2、r3 各有 1 窗被判复读循环（含循环的 cpER 30.0 / 30.4），v1 这两档没有；138 窗 3 次里只有 2 次，未深究。
- 无重叠文字 3.24 低于真实 4.45，说明素材是模型见过的语音（局限见 [02](../02-fidelity-check/README.md)），不能读成比真实容易。
- 重度重叠档没变，和只打断上下文的 near_shuffled 30.3、shuffled_far 44.5 一致，是重配音本身的属性（[07](../07-far-gap-probe/README.md)）。
- 对话统计基本不变（[synth-label-stats-120s-whole.json](../../tracking/synth-label-stats-120s-whole.json)）：AliMeeting 模板重叠 31.4%、附和 8.1 次/分、嵌入句 5.06 次/分。

## 代价：素材重复次数

合成占训练 29%、12000 步（每步 32 个样本），模拟放置算得每类素材被听的遍数：

| 配方 | 2 人会议 | 3 人会议 | 4 人会议 | AISHELL-5 2 人 |
| --- | --- | --- | --- | --- |
| v1（不用齐），AliMeeting | 26 | 28 | 24 | — |
| 用齐，AliMeeting | **73** | **74** | 11.5 | — |
| 用齐，AliMeeting 占比降到 15% | 38 | 38 | 5.9 | — |
| 用齐，AliMeeting + AISHELL-5 1:1 | 33 | 38 | 5.6 | 14.5 |

最终配置（用户决定）：AliMeeting + AISHELL-5 1:1，合成占 24.7%，排除 49 场问题会议（[../09-material-audit/README.md](../09-material-audit/README.md)）。遍数：AliMeeting 2 人 29.9、3 人 29.6、4 人 4.2；AISHELL-5 2 人 12.0、3 人 28.6、4 人 4.4。

对照：第二轮 I 组把真实会议压到每秒约 5 遍，判断依据是更高的重复曾被背熟。重配音后每遍的混合都不同，但单句音频相同，高重复的影响训练前测不出来。
