# 重配音为什么多报说话人（干预解码）

- 输入：[scripts/build_dev_split_probe.py](../../scripts/build_dev_split_probe.py) 生成的 6 个近讲变体（dev 里 2–3 人的 69 窗，`meeting_synth_fidelity@v3-20261101`）、起点模型 Emilia2 6000、[configs/probe-split.yaml](../../configs/probe-split.yaml)（vLLM 部署栈）。
- 完成条件：每个变体解码 3 次并合并，W&B 远端可查（`AmphionASR-zh-diar-synth-split-probe-001`，已 finished）。
- 执行：attempt 001 完成，exit 0。

## 起因

v1 重配音上模型拆人：无重叠句的归属额外错误里"拆人"占 2.6（真实语音各变体 0–0.25，[attribution-breakdown.json](../02-fidelity-check/attempts/002/artifacts/attribution-breakdown.json)）。2 人窗口 62%、3 人窗口 32% 多报，4 人窗口 1.5%（模型几乎不输出第 5 人）。多报比例随来源会议人数上升（2 人窗口：来源 2 人 22%、3 人 57%、4 人 73%，[fill-overcount-dose-alimeeting.json](../02-fidelity-check/attempts/002/artifacts/fill-overcount-dose-alimeeting.json)）。

## 结果（多报说话人的解码比例，69 窗 × 3 次）

| 变体 | 做法 | 多报 | 拆人（无重叠句） |
| --- | --- | --- | --- |
| real_gated | 真实语音，同一渲染（对照） | 4.3% | 0.13 |
| real_plus_leak | 真实语音 + 两个陌生人 −22 dB | 66.2% | 0.32 |
| unseen_src2 | 没见过的 Eval 语音，来源 2 人会议（47 窗） | 13.5% | 0.39 |
| unseen_src4 | 没见过的 Eval 语音，来源 4 人会议 | 45.4% | 2.56 |
| v1 重配音 | 训练素材（同窗口子集） | 52.2% | |
| thinned | v1 随机删句到与 no_foreign 同时长 | 39.1% | 2.11 |
| no_foreign | 同样的声音，去掉与窗口外的人重叠的句子 | 41.5% | 3.56 |

配对差 [95% bootstrap]：unseen_src4 − unseen_src2 +40.4 [+24.8, +56.7]；no_foreign − thinned +2.4 [−10.6, +15.5]；real_plus_leak − real_gated +61.8 [+48.8, +73.9]；v1 − unseen_src4 +6.8 [−7.7, +20.8]。

## 读法

- **与"模型记得训练会议"无关**：没见过的语音同样多报，且与 v1 差异在噪声内。
- **来源人数是真实因素**：三个数据源一致（AliMeeting 训练 22% / 73%，Eval 13.5% / 45%，AISHELL-5 来源 2 人 8%）。
- **按标注去掉窗口外的人的串音无效**，所以"串音"不能作为已证实的原因。`real_plus_leak` 只说明模型会转写直接混入的陌生人声（多出的字 97% 不在标注里），特征和 v1 不同（v1 多出来的字主要是标注里已有的话），所以 −22 dB 这个数不能直接外推到真实串音。
- **声音一致性不是原因**（[voice-consistency.json](attempts/001/artifacts/voice-consistency.json)，ERes2NetV2）：各来源人数的同人一致性 0.78–0.80；被拆走的句子略低（配对差 −0.05，61% 为负，Eval 素材 −0.02，47%）；最相似两人的相似度与多报没有单调关系。
- **机制未定**：候选是标注之外的串音残留，或对话缺人导致的上下文不连贯（`shuffled_far` 里打断上下文也让多报升到 14.5%，见 [07](../07-far-gap-probe/README.md)）。两者没有分开，修法（[08](../08-whole-conversations/README.md)）对两者都有效，所以不再为此加实验。
