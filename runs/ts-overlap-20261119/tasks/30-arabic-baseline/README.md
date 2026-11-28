# 30 阿拉伯语：11.17 基线

## 要回答的问题

交付线按"重叠场景 → 阿拉伯语 → 葡萄牙语"推进（用户 11-28 定的节奏）。加阿拉伯语之前，先量出 11.17 在阿语评测上的水平，作为后续阿语训练的对照和守门基线。

## 输入

- 模型：11.17 交付包，缓存名 `whitebox-merge-v18a-es116inc`（条目 `model-afea989da22b77adc0ce9365`）。服务栈和交付守门相同：[eval_lanes.sh](../../scripts/eval_lanes.sh)，pack-mtp `serve.sh`，不开 MTP。
- 评测集（AmphionEval MR !174，均为 `@v1-20261127`，由 audio-data-contract-d9 建）：
  - `asr/ar-asr-test`：CV 24.0 test、FLEURS ar_eg、MGB-2 test、Casablanca 8 种方言；
  - `asr/ts-asr-benchmark-ar` v1 / v2：v2 在 v1 基础上加了补充集各列；
  - `asr/ts-asr-crosslingual-ar`；
  - `asr/hotword-bench-ar`；
  - `asr/ts-asr-hotwords-ar`；
  - `asr/asr-babble-bench-ar`。
- 配置：[configs/ar-templates/ar/](../../configs/ar-templates/ar/)，取自 d9 的冒烟配置（`/workspace/workspace/mingdong/data/amphiondata/ar-benchmarks-20261127/smoke/`），只把 `samples_per_source` 改成 0（全量）。不发布。
- 打分：共享的 amphion-eval 环境是 0.13.3，没有这些评测集，也没有阿语打分规则（78c1583）。所以评测程序用固定快照 amphion-eval ca7d87f（原生打分 v6），装在 `/workspace/data/ts-overlap-20261119/ar-baseline/amphion-eval-0.15.0-ca7d87f`，通过 `AE_PATH` 放到 PYTHONPATH 最前面。共享环境不动。
- 作业：[job_ar_baseline.sh](../../scripts/job_ar_baseline.sh)，1 节点，4 路并行。输出 `/workspace/data/ts-overlap-20261119/task30-ar-r1117/`，评测记录在 `/workspace/data/ae-runs/whitebox-merge-v18a-es116inc/`。

## 完成条件

7 个评测都 rc=0、覆盖率 100%，W&B 远端 finished 且指标已上传。这是基线测量，不设达标门槛。

## 进展

- 2026-11-28：准备配置和固定快照，提交前通知其他会话。
