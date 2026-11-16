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
- **进展**：
  - 2026-11-15 排在 ACP 任务 `pt-ehwnc3gs` 上。节点被抢占后改到 `pt-kva8cue7`，并拆成 6 组（`configs/eval-grpo-avg3-mtp-{b,c,d}-{ts,asr}/`）。
  - 2026-11-16 04:11Z 开始，05:59Z 结束，6 组退出码都是 0。
  - 9 项完整、无失败样本。W&B 远端 9 个 run 都是 finished，已上传到 whai。
- **结果**：鼎桥场景 TS 明显退步：CER 3.35 → 9.24，目标不在场的误出字 16 → 48 条，只出现在“分开讲话”切片。其余 8 项持平，或差异在用户已接受的范围内。数字和 run 见实验 README 的“05 结果”，定位见 08。
