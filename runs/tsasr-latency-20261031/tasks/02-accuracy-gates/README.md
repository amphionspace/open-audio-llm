# 02 准确率闸门

- 输入：AmphionEval `open-audio-llm/ts-asr-benchmark@v12-20261022` 全量（26,750 条，enroll_seconds 5），10.28 版本权重。
- 方法：`scripts/run_gate.sh <suffix|fp8|prompt5s>` 起服务变体并跑 v12；prompt5s 经 `scripts/prompt_rewrite_proxy.py` 把训练用 "3-second" 指令换成线上 audiollm-go 发送的 "5-second"。
- 对照：`model_name=ts-robust-054-avg3` 的 v12 结果；batch 不确定性的参照是交付包自检（同权重 50 条在场判断不同）。
- 完成条件：三项各有 v12 全量结果，与对照逐条比较 CER、漏识、误输出。
