# 04 只解冻音频塔（对照组）

## 作用

用来区分"解冻音频塔"本身和"干净原句 KL"各自的作用。

- 配置与 [03](../03-clean-kl/README.md) 完全相同，只是数据配置里没有 `clean_teacher_kl_weight`。
  - 配置：[../../configs/train-control.yaml](../../configs/train-control.yaml)，数据配置 [../../configs/train-data-control.yaml](../../configs/train-data-control.yaml)。
- 历史上 v95 之前音频塔一直随文本交叉熵更新，重叠转写没动。所以预期本组的重叠转写基本不变。
- 起点（11.17）和数据都与那些轮次不同，所以仍要跑这一组。

## 完成条件

同 03。评测与 03 配对比较。

## 进展

- 2026-11-20：冒烟（ACP `pt-sj78mv90`，与 04 同节点各 4 卡，40 步，脚本 `scripts/run_smoke.sh`）。
  - attempt 001 失败：main 需要更新版的 audio-data-contract（缺 `source_key_specs`），code-v37 的那份太旧。改为快照 `/workspace/data/audio-data-contract` 211680bc，与本机单元测试用的是同一版本。
  - attempt 002 失败：数据配置的 `batching.accumulation` 必须等于 `--gradient_accumulation_steps`。两份数据配置都改成 6。
  - attempt 003 执行完成（exit 0），W&B 远端核验：`AmphionASR-ts-overlap-{cleankl1,control}-audunfrozen-from1117-v96data-smoke40-4gpu-codev1-003`。核对结果：
    - 第 1 步两组 `ce_ts_positive` 都是 0.21349347，`ce_asr_no_kl`、`ce_ts_negative` 也逐位相同，学生样本流一致；
    - 实验组 `clean_kl` 为 0.16–0.19，每个微批约 2 条样本带干净原句；
    - 两组可训练的音频参数都是 305,461,248；
    - 速度（40 步累计平均）实验组 3.24 秒/步、对照组 2.92 秒/步，显存峰值 73.1 / 71.8 GiB。
  - 执行记录：`whai:open-audio-llm/runs/ts-overlap-20261119/tasks/{03-clean-kl,04-control}/attempts/00{1,2,3}`。
- 正式 6000 步训练尚未启动，等用户确认。
