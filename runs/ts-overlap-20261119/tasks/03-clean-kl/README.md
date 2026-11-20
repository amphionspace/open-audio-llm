# 03 干净原句老师 KL（实验组）

## 假设

重叠时的转写错误一直降不下来，是因为训练里没有直接的信号告诉模型"重叠时该转写成什么样"：只有参考文本的交叉熵。同一模型对同一句话单独录音时只错 3.1%。所以用它看干净原句时的输出分布当老师，对混音输入做 token 级 KL，信号比只用参考文本密得多。

## 唯一主要变量

与 [04 对照](../04-control/README.md) 只差数据配置里的 `objective.clean_teacher_kl_weight: 1.0`。

## 输入

- **配置**：[../../configs/train-cleankl.yaml](../../configs/train-cleankl.yaml)，数据配置 [../../configs/train-data-cleankl.yaml](../../configs/train-data-cleankl.yaml)。
- **起点**：11.17（ae 模型缓存 `whitebox-merge-v18a-es116inc`）。
- **配方**：沿用 v96，数据同 v94/v96，含西语普通识别 KL 2.0。
- **音频塔**：按 v94 解冻，只冻卷积前端，与 LLM 同一学习率 2e-6。
- **规模**：4 卡 × 每卡 8 条 × 累积 6，全局 192，共 6000 步。与 04 在同一节点各用 4 卡。
- **KL 作用的样本**：目标在场的在线混音 TS 样本（单句干净目标与场景池）。
  - 老师是模型自己，看"同一注册 + 干净原句"，不传梯度。
  - 干净原句与混音同一时间窗、同一增益，不额外消耗随机数，所以 03 和 04 的学生样本完全相同。
  - 真实录音窗口（有 `target_speaker`）没有干净原句，不做 KL。

## 完成条件

- 冒烟：40 步跑通，`clean_kl` 和 `clean_rows` 有值，速度可接受。
- 正式：6000 步执行完成，W&B 远端核验。
- 评测与 04 配对比较，口径用 `scripts/ts_gate_breakdown.py`：
  - 中文 TS v12 守门 12.88 / 371 / 369，外加重叠分箱转写错误；
  - 英文 TS 误报（11.17 是 655）；
  - 中文普通识别不劣于 11.06。
- 要回答的问题：重叠 ≥ 0.9 的转写错误（11.17 是 15.64）是否明显低于 04。

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
- 2026-11-20 07:10 UTC：用户确认后启动正式训练，attempt 004（ACP `pt-7h66hzg8`，与另一组同节点各 4 卡）。
  - 脚本 `scripts/run_train_eval.sh`：训练 → 第 5000/5500/6000 步平均（`scripts/avg3_cache.sh`）→ 评测（`scripts/eval_lanes.sh`：中文 TS v12、英文 TS、中文普通识别、客户 TS）。
  - W&B 远端已读回训练指标：`AmphionASR-ts-overlap-{cleankl1,control}-audunfrozen-from1117-v96data-6k-4gpu-codev1-004`。
  - 速度：实验组约 3.05 秒/步，预计 5 小时；对照组约 2.69 秒/步，预计 4.4 小时。
