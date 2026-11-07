# 02 TS 流式回放：MTP 的预览滞后

- 输入：v16 avg3 + `shared/draft-v16avg3`；服务包 `shared/pack-mtp`；配置 `configs/stream-lane{0,1,2}.yaml`（base / MTP k2 / MTP k3，各占一张卡同时跑）；参数同 mtp-draft-head-20261031 的 stream-final，提示词换成新 TS 提示。
- 入口：`scripts/run_stream_bench.sh attempts/<编号>`；汇总 `scripts/summarize_stream.py` + `scripts/show_stream.py`。
- 完成条件：三档并发（1/16/48，各 300 s）都有结果；W&B 远端核验。

| 执行 | 结果 |
|---|---|
| 001 | 完成；W&B `j8idjst8`、`e5kvhyp2`、`yi9xk4y4` 远端 finished（`attempts/001/wandb-verification.json`） |

预览滞后 p50 / p95 与 final 延迟 p50 / p95（ms）：

| 并发 | 无投机 | MTP k2 | MTP k3 |
|---:|---|---|---|
| 1 | 89/188，111/363 | 65/136，84/214 | 62/137，78/190 |
| 16 | 103/259，119/368 | 73/180，98/212 | 69/166，91/194 |
| 48 | 172/470，199/641 | 117/386，146/363 | 107/321，127/304 |

流式下 k3 比 k2 更快；CER 三者相当（片段切分随时间变化，不逐条对比）。
