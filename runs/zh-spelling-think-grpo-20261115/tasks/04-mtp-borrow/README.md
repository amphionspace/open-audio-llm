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
- **进展**：
  - 2026-11-16 在节点上第一次启动失败：普通识别服务挂不上草稿，由 open-audio-llm #161 修复。
  - 之后改在本机 GPU 5 上跑（执行记录 `local-serve-*`），04:06Z 完成。
- **结果**：k=2 时接受长度只有 1.64（think 关）和 1.49（think 开），远低于草稿在 v16 avg3 上的 2.81。分数在数值噪声范围内不变。详见实验 README 的“04 结果”，后续补训见 06。
