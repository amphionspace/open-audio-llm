# 07 avg3 自己的草稿头：接受长度与是否无损

- **输入**：
  - 模型：avg3（ae 模型缓存 `model-377c0d630417ca1fc281e847`）。
  - 草稿：06 训练的 EAGLE-3 草稿头，`tasks/06-train-draft/attempts/002/artifacts/training/checkpoint-10000/eagle`，vLLM `method: eagle3`。
- **配置**：`configs/eval-grpo-avg3-eagle3-k{3,2}/`，由 `scripts/make_draft_accept_configs.py <草稿目录>` 从 04 的配置生成，只换草稿、GPU、端口、服务名和记录目录。
  - k=3 是 mtp-draft-head-20261031 测速后选定的交付设置；k=2 与 04 直接对照。
  - 两路同时跑在交付节点上：k=3 用 GPU 0/1，k=2 用 GPU 2/3。
- **入口**：`scripts/run_mtp_check.sh configs/eval-grpo-avg3-eagle3-k<k> <W&B 代理> <ae> context-correction context-correction-think`，由 inbox 脚本 `60-draft-accept.sh` 在 06 退出码为 0 后启动。
- **完成条件**：
  - 四项评测完整；
  - 每项前后的 `/metrics` 都已保存；
  - 接受长度按普通识别 / TS、think 关 / 开分开算出，与 04 和 10.28 草稿头（k=2 2.72 / 2.66，k=3 3.43 / 3.31）对照；
  - 输出与不开 MTP 的 run（`4bf6261c`、`e0b3a6b5`）逐条比对。
- **进展**：2026-11-16 19:21Z 在节点 `pt-1blli2zm` 上开始（作业 `45-draft-accept.sh`），19:25Z 四项都完整结束。
- **结果**：k=3 MAL 3.61–3.69，k=2 2.81–2.85，都高于 10.28 交付草稿头。输出与不开 MTP 的 run 逐条比对，差异在数值噪声范围内。详见实验 README 的“07 结果”。
