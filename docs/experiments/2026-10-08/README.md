# 2026-10-06 至 10-08：8×A800 续训、数据质检与 180 秒会议评测

**结论：MOSS-Transcribe-Diarize 仍全面领先。** 这一阶段的三轮训练都没有超过起点（LoRA 002 / checkpoint-8000）；评测体系补齐后，确认我们最大的短板是 6 人以上、英文会议和 180 秒片段上的复读循环，不是数据标注质量。本机训练已于 10-08 按用户要求暂停，后续训练改用提交方式。

接续 [2026-10-06 归档](../2026-10-06/README.md)。实验目录 `runs/unified-diarization-asr-20261001/`，各任务 README 有逐次执行细节；机器可读结果见 [results.json](results.json)。

## 对标目标

| 评测 | MOSS（官方 vLLM） | 我们最好的点 | 差距（95% 区间） |
| --- | ---: | ---: | --- |
| 固定集 338 条 cpCER | **15.53%**（原生 FP32 15.39%） | 20.44%（LoRA 8000） | +4.91 [3.45, 6.25] |
| 180 秒评测集·中文会议 | **16.43%** | 32.51%（去复读约 22%） | +16.1 [7.4, 25.1] |
| 180 秒评测集·英文会议 | **22.72%** | 61.62% | +38.9 [32.3, 47.4] |
| 180 秒评测集·6–10 人合成 | **19.60%** | 85.88% | +66.3 [61.5, 70.8] |

## 执行记录与结论

| 任务 / 执行 | 做了什么 | 结果 | 结论 / 决定 |
| --- | --- | --- | --- |
| localize-data 004 | 原机数据迁到本机：catalog 原样，路径用硬链接镜像 | 4,023,765 条，SHA256 全部一致 | 之后所有训练的数据基础 |
| local-training 003 | LoRA 002 从 8000 步 8 卡续训 | 学习率实际 5e-6（应为 1.51e-5），约 8040 步停止，未留 checkpoint | 根因：ms-swift 4.5 绕过调度补丁；已修复（PR #31） |
| local-training 004 → 005 | 修复后续训到 10000，再无训练内评测续到 25500 | 固定集 25500：21.17%，比 8000 差 0.73（p=0.055） | 8000 之后会议集变差，真实会议重复读取约 8 次，疑似过拟合 |
| evaluate-a800 002 | A800 同机固定集：1000 / 4000 / 8000 / 25500 | 21.84 / 20.65 / **20.44** / 21.17% | 8000 为当前最好点 |
| sft-25500 001 | 以 25500 合并模型为基座，新 LoRA，每卡 4 条不累积 | 500–6000 步固定集 21.1–21.8%，均未显著好于基座 | 用户叫停；改为先分析原因 |
| 原因分析 | 错误分解：识别 vs 说话人归属 | 低重叠片段归属损失约 0.3 点，差距几乎全在识别（11.9% vs MOSS 9.9%） | 方向：远场识别、全参数、加噪 |
| audit-meeting-windows / score-training-losses | 真实会议窗口逐条对照原始标注；8000 模型逐条 loss | 标注一致；隔离 R1021_M1947（音频与标注不匹配）；dense 合成档 loss 中位数 0.73 | 数据质量不是主因；清理后进入全参训练 |
| prepare-full-training-data 001 | 排除 2,814 窗口，留出 18 场会议作选模集，去 dense，合成与单人 ASR 加 MUSAN 噪声 | 选模集 245 段 4.9 h | — |
| full-8000 001 | 8000 合并模型全参数训练（编码器 5e-6） | 选模集 13.66% → 15.01 / 15.71 / 16.62%（500 / 1000 / 2000 步），持续变差 | 2500 步暂停（不再用本机训练）；退化原因未定位 |
| build-meeting-benchmark 003 | 180 秒部署长度评测集，424 段 21.2 h | 中文 / 英文真实会议 + 6/8/10 人合成 | 后续标准评测之一 |
| evaluate-moss-* / score-* | MOSS 官方 vLLM 服务；同口径汇总 | 见上表 | MOSS 目标重申为 15.53% |

## 已确认问题与待办

1. **6 人以上**：训练数据 6–7 人仅占 0.38%，没有 8 人以上；需要 6–12 人的合成数据（MOSS 为 2–12 人）。
2. **英文会议**：AMI cpER 55–68%（去复读），说话人数准确率 < 30%；训练没有英文真实会议。
3. **复读循环**：180 秒片段上 12 / 424 段复读至 token 上限，MOSS 为 0；部署需解码约束或训练修正。
4. **全参训练退化**：选模集每千步约升 1 个百分点；留出会议曾参与起点训练，需用固定集或 180 秒评测集判定是否真实退化。
5. **评测局限**：合成档噪声与训练加噪同源（MUSAN）；真实会议切点对齐说话空隙；无 8 人以上真实会议。

## 修复与合入

PR #31（游标迁移、调度器、W&B 重试）、PR #33（catalog 2.0 / AmphionEval 0.6.0）已合入。本分支另含：`exclude_records` 与 `noise_exclude_datasets`、全参 checkpoint 评测、180 秒评测集与 MOSS 基线脚本。

## 留存位置

- Git：配置、脚本、任务 README、本目录汇总 JSON（`meeting-benchmark-summary.json`、`fixed-set-*.json`、`meeting-benchmark-composition.json`）。
- 对象存储 `whai:open-audio-llm/runs/unified-diarization-asr-20261001/`，路径与本地一致：全部执行记录、评测集音频与清单、原始输出与逐条打分、以及 local-training 005 / checkpoint-25500、sft-25500 / checkpoint-7000、full-8000 / checkpoint-2500（含优化器）。清单与排除规则见 [upload-manifest.json](upload-manifest.json)；合并模型可由 adapter 加基座重新合并，未上传。
- W&B（`1016097967-amphion/open-audio-llm`）：`unified-diarization-local-lora-003/004/005`、`unified-diarization-sft-25500-001`、`unified-diarization-full-8000-001`、`unified-diarization-a800-eval-002`、`unified-diarization-select-eval-002`、`unified-diarization-meeting180-summary-001`、`moss-transcribe-diarize-vllm-fixed-001`、`moss-transcribe-diarize-vllm-meeting-benchmark-001`。
