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
  - 2026-11-21 08:27 UTC 训练、平均、四套评测全部执行完成（exit 0），随即停掉 `pt-3ml2y432`。
  - 第 5000/5500/6000 步平均：ae 模型缓存 `ts-overlap-frozenllm-002-avg3`（`model-5f1fc2643cc9475383fc91fd`）。
  - 评测 run（W&B 远端均为 finished，v12 run 带 29 个 `breakdown/*` 键）：v12 `run-20261121T072002Z-9c26b955`，英文 TS `run-20261121T072935Z-ecebae0f`，普通识别 `run-20261121T072047Z-b363db72`，客户 TS `run-20261121T082558Z-61cf9a44`。

## 结果

**不达标。** v12 字错率 13.54、误报 419、漏识 372（守门 12.88 / 371 / 369）。各项数字和与 07 的配对比较见 [07 的结果](../07-clean-feature/README.md#结果)。

- 冻结 LLM 没有避免从 11.17 接着 SFT 的退化：每个重叠档都比 11.17 差，误报 + 漏识从 699 回到 791，中文普通识别 4.39 → 4.56。
- 和 04（放开 LLM）相比：重叠 ≥ 0.9 转写错误 16.17 → 15.71，漏识 439 → 372，英文 TS 误报 805 → 660（守住）；但中文误报 366 → 419。

## 证据

- `whai:open-audio-llm/runs/ts-overlap-20261119/tasks/08-frozen-llm/attempts/002/`：执行记录、日志、各检查点；`artifacts/eval/breakdown-v12.json`、`paired-v12-1117-control-frozenllm-feat.txt`。
- 评测输出：`whai:open-audio-llm/runs/ts-overlap-20261119/ae-runs/ts-overlap-frozenllm-002-avg3/runs/`。
