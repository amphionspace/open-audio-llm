# 03-analyze：汇总与配对比较

## 输入

- 配置：[`configs/analyze.yaml`](../../configs/analyze.yaml)，脚本 [`scripts/analyze.py`](../../scripts/analyze.py)。
- 数据：01 执行 002、003 与 02 执行 001 的作业 `result.json` 及其中记录的逐条打分（`meeting-scores.jsonl`，补分时为 `meeting-scores-rescored.jsonl`；本次没有补分）。

## 计算

- 每个（模型、设置、重复、划分）：AmphionEval 摘要中的 cpER、排除复读 cpER、复读条数、DER、人数准确率。
- 共同子集 cpER：同一模型所有执行（含重跑）都未复读的样本。
- 复读配对：每条复读样本在哪些执行中复读；各执行相对 `default/r1` 修复、新增、仍复读的条数（只在核心 928 条上算，固定集只有部分执行跑过）。
- 重跑波动：`start`、`realmix1500` 默认解码 3 次的复读条数（值、范围、样本标准差）和各划分 cpER / 排除复读 cpER / 共同子集 cpER 的范围。

## 结果

执行 001：completed（0）。产物 `attempts/001/artifacts/analysis.md`（全部表格与逐条复读样本）、`analysis.json`、`analysis-metrics.json`；W&B https://wandb.ai/1016097967-amphion/open-audio-llm/runs/decode-loop-study-20261009-analysis-001 （finished）。

重跑波动（默认解码 3 次）：

| 划分 | start 复读 | start 排除复读 cpER 极差 | realmix1500 复读 | realmix1500 排除复读 cpER 极差 |
|---|---|---|---|---|
| 核心 928 条合计 | 48 / 46 / 52（σ 3.06） | — | 40 / 35 / 38（σ 2.52） | — |
| AliMeeting dev | 0 / 1 / 2 | 0.06 | 2 / 1 / 2 | 1.07 |
| AISHELL-4 test | 0 / 0 / 0 | 0.31 | 0 / 0 / 0 | 0.98 |
| AliMeeting test | 7 / 7 / 8 | 1.63 | 8 / 8 / 10 | 0.42 |
| NOTSOFAR dev | 19 / 19 / 20 | 3.81 | 7 / 11 / 7 | 1.89 |
| AMI test | 5 / 4 / 4 | 1.41 | 8 / 5 / 7 | 0.92 |
| CHiME-6 dev | 17 / 14 / 17 | 2.18 | 15 / 10 / 12 | 2.86 |

复读过的样本中，三次都复读的 `start` 8/106、`realmix1500` 16/70；其余只在 1–2 次中复读。解码设置对比与结论见实验 [README](../../README.md)。
