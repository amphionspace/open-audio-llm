# 08 冻结 LLM、只训音频塔（对照组）

## 作用

用来区分"冻结 LLM 只训音频塔"本身的作用和表示损失（任务 07）的作用。

- 与 [07](../07-clean-feature/README.md) 完全相同，只是数据配置里没有 `clean_teacher_feature_weight`。
  - 配置：[../../configs/train-frozenllm.yaml](../../configs/train-frozenllm.yaml)；数据配置 [../../configs/train-data-frozenllm.yaml](../../configs/train-data-frozenllm.yaml)。
- 和任务 04 相比，它冻结了 LLM，音频塔学习率 5e-6，训练文本用终版。可以顺带看冻结 LLM 能否避免 04 在 v12 上的退化。

## 完成条件

同 07。评测与 07 配对比较。

## 进展

- 2026-11-21：冒烟 attempt 001 与 07 同节点，执行完成，核对结果见 [07 的进展](../07-clean-feature/README.md#进展)。
- 2026-11-21 02:33 UTC：正式训练 attempt 002，与 07 同节点，ACP `pt-3ml2y432`。
  - W&B 远端已读回训练指标：`AmphionASR-ts-overlap-llmfrozen-aud5e-6-from1117-v96final-6k-4gpu-codev2-002`。
