# 04 avg3 借 MTP 草稿：接受长度与是否无损

- **输入**：
  - 模型：avg3（ae 模型缓存 `model-377c0d630417ca1fc281e847`）。
  - 草稿：v16 avg3 联合训练的 MTP，`/workspace/workspace/mingdong/repo/oal-joint-mtp/runs/joint-mtp-spec-20261107/shared/draft-v16avg3`，EAGLE 提议器，k=2。
- **配置**：`configs/eval-grpo-avg3-mtp/`，与 `configs/eval-grpo-avg3/` 只差 GPU、端口、服务名和 `--speculative-config`。
- **入口**：`scripts/run_mtp_check.sh configs/eval-grpo-avg3-mtp <W&B 代理> <ae> context-correction context-correction-think`。
- **完成条件**：
  - 两项评测完整；
  - 每项前后的 `/metrics` 都已保存（`configs/eval-grpo-avg3-mtp/logs/metrics-*.txt`）；
  - 接受长度已按普通识别 / TS、think 关 / 开分开算出；
  - 输出已和不开 MTP 的 run 逐条比对。
- **进展**：2026-11-15 已排在 ACP 任务 `pt-ehwnc3gs` 上。
- **结果**：待填。
