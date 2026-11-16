# 06 avg3 的 EAGLE-3 草稿头

- **输入**：
  - 骨干：avg3（冻结），ae 模型缓存 `model-377c0d630417ca1fc281e847`；
  - 数据：`configs/train-data-draft.yaml`，即第二轮 think 数据配置原样；
  - 代码：`shared/code-v3`。
- **配置**：`configs/train-draft.yaml`，由 mtp-draft-head-20261031 的正式配置复制。EAGLE-3，3 步展开，8 卡，全局 batch 192，10000 步，学习率 1e-4，余弦降到 1e-5。
- **入口**：`scripts/train_draft.sh <runner 目录> <05 的任务名…>`，等 05 结束再开始。
- **完成条件**：
  - 训练正常结束；
  - `checkpoint-10000/eagle/` 已上传；
  - 用它复测接受长度（同 04），并与 04 的结果比较。
- **进展**：2026-11-16 已排在 ACP 任务 `pt-kva8cue7` 上。
- **结果**：待填。
