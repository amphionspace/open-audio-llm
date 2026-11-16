# 05 avg3 交付评测（开 MTP）

- **范围**：11.06 交付包 `validation/results/results.md` 里 avg3 还没测的 9 项。已测的 8 项守门和上下文纠偏见实验 README。
- **配置**：`configs/eval-grpo-avg3-mtp-{b,c,d}/`，由 `scripts/make_gate_configs.py <目录名> --model grpo-avg3-mtp` 生成。
  - 每项复制交付版那次 run 的配置，只换模型、服务和输出目录。
  - 服务加 MTP 草稿，k=2，同 04。
- **分组**（每组一对服务，`scripts/run_eval.sh`）：
  - b（GPU 2/3）：`gate-first-other`、`gate-babble4`、`gate-police-single`；
  - c（GPU 4/5）：`gate-ts-en`、`gate-hotword-bench`、`gate-scene-asr`；
  - d（GPU 6/7）：`gate-ts-hotwords`、`gate-scene-ts`、`gate-badcase`。
- **完成条件**：
  - 9 项 run 完整（`ae runs show` 的 coverage 全满）；
  - 与交付版同一次 run 做配对比较；
  - W&B 远端已核验；
  - 结果写回实验 README。
- **进展**：2026-11-15 已排在 ACP 任务 `pt-ehwnc3gs` 上，等 04 的 W&B 转发确认后与 04 同时开跑。
- **结果**：待填。
