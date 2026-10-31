# 02 准确率闸门

- 输入：AmphionEval `open-audio-llm/ts-asr-benchmark@v12-20261022` 全量（26,750 条，enroll_seconds 5），10.28 版本权重。
- 方法：`scripts/run_gate.sh <suffix|fp8|prompt5s>` 起服务变体并跑 v12；prompt5s 经 `scripts/prompt_rewrite_proxy.py` 把训练用 "3-second" 指令换成线上 audiollm-go 发送的 "5-second"。
- 对照：`model_name=ts-robust-054-avg3` 的 v12 结果；batch 不确定性的参照是交付包自检（同权重 50 条在场判断不同）。
- 完成条件：三项各有 v12 全量结果，与对照逐条比较 CER、漏识、误输出。

## 结果（v12 全量，评分协议 v2；`results/compare.json`）

| | CER | 漏识 | 误输出 | 在场判断翻转 |
|---|---|---|---|---|
| 10.28 原始（rescore v2） | 13.07 | 2.85 | 3.28 | — |
| 同权重重跑（交付包自检，噪声） | 13.02 | 2.88 | 3.34 | 50 |
| suffix 投机解码 | 13.04 | 2.84 | 3.35 | 40 |
| FP8 权重量化（Marlin） | 13.14 | 2.90 | 3.26 | 80 |
| 线上 "5-second" 指令 | 13.06 | 2.86 | 3.33 | 45 |

`run-20261031T173212Z-ee3d536a` 无效（AmphionEval 按权重复用了 fp8 的预测），未上传 W&B；之后所有闸门配置加 `reuse: false`。
