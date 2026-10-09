# 02-fixed-best：固定 338 条集用 rp1.1

## 输入

- 配置：[`configs/fixed-best.yaml`](../../configs/fixed-best.yaml)（驱动、模板、核验与 01 相同）。
- 设置：`repetition_penalty: 1.1`。01 中它的复读最少（核心 928 条 18–30 条），作为“最优设置”代表；默认解码的固定集结果在 01 各模型 `default/r1`。
- 模型：同 01 的 5 个；评测集：固定 338 条中文集（AISHELL-4 200 + AliMeeting 138）。

## 完成条件

5 个作业 `result.json` 存在、解码核验通过（每个服务 338 个请求，`repetition_penalty` 只见 1.1）、W&B 远端核验完成。

## 结果

执行 001：completed（0），5 个作业全部通过核验；W&B https://wandb.ai/1016097967-amphion/open-audio-llm/runs/decode-loop-study-20261009-fixed-best-001 （finished），5 个评测 run 均为 finished。

| 模型 | AISHELL-4 cpER default → rp1.1 | AliMeeting cpER default → rp1.1 | 复读 default → rp1.1 | 共同子集变化（AISHELL-4 / AliMeeting） |
|---|---|---|---|---|
| start | 18.82 → 19.85 | 25.95 → 23.36 | 2 → 0 | +1.03 / +1.13 |
| realmix500 | 40.26 → 20.46 | 22.89 → 29.87 | 1 → 1 | +1.37 / +2.05 |
| realmix1500 | 19.53 → 19.13 | 30.81 → 22.36 | 3 → 1 | +0.50 / +1.91 |
| lrhalf500 | 18.13 → 19.30 | 25.93 → 22.31 | 1 → 0 | +1.17 / +0.98 |
| lrhalf2000 | 19.00 → 19.11 | 21.38 → 22.00 | 1 → 0 | +0.12 / +0.62 |

含复读的 cpER 主要随哪条样本复读而跳动；在同一批未复读样本上，rp1.1 在 10 个对比中全部变差 0.1–2.1 点。
