# 清洗事件数据对照实验

两组均已完成 1000 步训练；固定集评测失败，尚无最终优劣结论。合成补充后达到 549 条、11.85 小时、645 个源说话人。声纹阈值为实验规则，未做校准；存疑项隔离，不再发起人工听审。

| 任务 | 执行状态 | 质量状态 |
|---|---|---|
| [扩充清洗来源](tasks/01-source-expansion/README.md) | 已完成 | 待核实 |
| [声纹筛查](tasks/02-identity-screening/README.md) | 已完成 | 待核实 |
| [合成与补充](tasks/03-synthesis/README.md) | 已完成 | 补充后自动门槛通过 |
| [准备对照训练数据](tasks/04-prepare-training/README.md) | 已完成 | 待核实 |
| [训练旧数据对照组](tasks/05-train-control/README.md) | 已完成 | 待核实 |
| [训练新数据实验组](tasks/06-train-treatment/README.md) | 已完成 | 待核实 |
| [固定集评测](tasks/07-evaluate/README.md) | 失败，待重试 | 待核实 |

两组共享起始权重、训练设置和固定 338 条评测，只替换 35% 的事件数据。结果只能归因于来源质量与合成流程的整体变更。

运行或预览：

```bash
open-audio-llm experiment run --config runs/clean-events-ab-20260928/experiment.yaml --dry-run
open-audio-llm experiment run --config runs/clean-events-ab-20260928/experiment.yaml --task evaluate
```

[原始方案](shared/plan.json)、[原始概览](shared/history/README.md)、[迁移核验](tracking/migration-verification.json)。原始总控中的“未开始训练”与训练退出和 W&B 记录冲突，保留原文并标记待核实；不凭 PID 文件判断运行中。
