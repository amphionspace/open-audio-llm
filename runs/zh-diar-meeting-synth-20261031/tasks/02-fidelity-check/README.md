# 起点模型在保真度变体上评测

- 输入：attempt 002 生成的 5 个变体（`meeting_synth_fidelity@v1-20261031`）、起点模型 emilia2 6000、
  [configs/probe-fidelity.yaml](../../configs/probe-fidelity.yaml)（vLLM 部署栈，与差距诊断同一套服务与打分）。
- 完成条件：5 个变体各解码 3 次并合并，W&B 远端可查；再用 `scripts/overlap_breakdown.py` 按重叠程度拆分。
- 判定（自动，不设人工听审）：syn_acoustic 和 syn_fill_far_* 对照真实远场 far_mean（cpER 21.42，无重叠 6.2，重度重叠 35.0）；
  syn_fill_near_* 对照近讲 near_gated（16.35，4.3，25.1）。旧合成数据 sim_train 为 26.13 / 11.0 / 34.5。差距要超过解码噪声（±0.5）才算数。
- 进展：
  - attempt 001 失败（exit 2）：`shared/catalog-fidelity/roots.json` 里目录根是相对路径，AmphionEval 只接受绝对路径。
    已改 `scripts/build_dev_fidelity.py` 写绝对路径，并把现有 roots.json 换成同样的绝对路径（音频与标注未变）。
  - attempt 002 于 2026-11-01T06:19Z 启动，06:34Z 起 vLLM 正常解码。

## 结果（attempt 002，3 次解码合并）

| 变体 | cpER | 无重叠 | 重度重叠 | 说话人数正确率 | 对照 |
| --- | --- | --- | --- | --- | --- |
| syn_acoustic | 19.6–20.0 | 6.3 | 35.9 | 0.93 | far_mean 21.42 / 6.2 / 35.0 |
| syn_fill_near_alimeeting | 22.4–23.4 | 6.2 | 30.9 | 0.67 | near_gated 16.35 / 4.3 / 25.1 |
| syn_fill_far_alimeeting | 30.4–30.9 | 10.2 | 44.1 | 0.65 | far_mean |
| syn_fill_near_aishell5 | 60–63（无循环 37.5，循环 6–9 窗） | 14.4 | 48.6 | | near_gated |
| syn_fill_far_aishell5 | 56–65（无循环 43） | 22.6 | 59.7 | | far_mean |

- 声学通过：真实 dev 语音经合成房间、音量与噪声后，各重叠档位与真实远场一致（略易约 1.5 点）。
- 重配音偏难，主要多在说话人归属：无重叠语句的归属额外错误 near +5.6（syn_acoustic +1.3），文字 6.2（near_gated 4.3）。
  原因未定位。"串音是原因"的假设已被否定，见 [../05-fidelity-check-clean/README.md](../05-fidelity-check-clean/README.md)；
  模型多数说话人（说话人数正确率 0.67，对 syn_acoustic 0.93）主要是把一个人拆成两个。
  已定位并修复，见 [../06-split-probe/README.md](../06-split-probe/README.md)、[../08-whole-conversations/README.md](../08-whole-conversations/README.md)：差距来自"只抽来源会议里的部分人"，而不是模型记得训练会议。上一句"串音已被否定"的依据（只统计 ≥3 字、看文字来源）不足以排除来源人数效应。
- AISHELL-5 与素材重叠比例关系不大（三档 13.8 / 14.5 / 15.9%），素材语句内信噪比与 AliMeeting 相当（约 32 dB）。全量统计
  （[overlap-breakdown.json](attempts/002/artifacts/overlap-breakdown.json)）里差距主要在文字本身：无重叠语句文字错误 14.4%
  （AliMeeting 6.2%），说话人归属的额外错误 +6.0（AliMeeting +5.6）。原因未定位，标注是否准确也未核实。此处早先写的
  "文字基本对得上、主要错在并人"只来自一个窗口，已撤回。
  2026-11-01 抽查文字错误最大的句子：主要是模型听错方言口语（邢台一带），标注全部为人工且与方言吻合，未见标注错误（只看文字，未听音频）。
- 远场文字差距已解释：约 2.7 / 3.9 点来自打断上下文在噪声下放大，见 [../07-far-gap-probe/README.md](../07-far-gap-probe/README.md)。

## 这次对照的局限

- 起点模型的训练数据含 AliMeeting 训练会议（远场、通道视图、近讲转远场）和 AISHELL-5 训练会议（远场），素材语音它都见过；
  真实对照 near_gated / far_mean 是没见过的 dev。所以重配音的错误率偏乐观，"干净素材与真实同水平"不能算保真度通过的证明。
- 串音电平（源耳麦音轨，他人独说相对本人独说）：AliMeeting 10 轨中位 −22.3 dB（p10 −29.4，p90 −20.8），AISHELL-5 19 轨中位
  −23.3 dB（p10 −31.2，p90 −11.7），见 tasks/03-data-check attempt 001 日志；该 attempt 的逐窗部分按计划未跑完。
