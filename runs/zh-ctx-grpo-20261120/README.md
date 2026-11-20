# 上下文纠偏的 GRPO 计划（2026-11-20 起）

**状态**：准备中，还没有训练。目标表和挖题规则已备好。起点要等上下文纠偏实验的第 4 问定下场景 TS 的修法，再确定。

**要回答的问题**：在 GRPO 里加一份上下文纠偏计划，能不能让 think 关时真实录音上的显式说明题多改对，同时守门和场景 TS 都不变差？
- Emilia 显式说明题现在 think 关只有 44%。
- 主要失败是听到了说明，却没有把它关联到前面的目标字。这是行为问题，奖励可以直接优化。

**依据**：
- 上下文纠偏实验 `runs/zh-ctx-clue-target-20261117`（分支 `feat/ctx-clue-target-think`）README 的“真实录音失败归因”和“下一步：GRPO 上下文纠偏计划”。
- 调研记录 `docs/research/2026-11-20-context-correction-related-work.md`。外部的主流做法是 SFT 之后做 RL，奖励给关键词加权，并惩罚多改。
- 奖励 `qwen3_asr_context` 和数据配置项 `grpo.context_targets` 已合入 main（#170），说明见 `docs/guides/training.md` 的 Qwen3-ASR GRPO。

## 设计

- **起点**：修好场景 TS 的上下文纠偏 SFT checkpoint，等第 4 问。
- **回放**：用起点那轮 SFT 的配方，即清洗后的转写版本，含合成说字数据。
  - 交付版那一轮 GRPO 的数据配置沿用了更早的配方，用的是未清洗的版本，也没有合成说字数据。本实验不沿用它。
  - 不沿用的原因：回放的参考答案必须和起点学过的写法一致，否则准确率奖励会把改对的字往回拉。按回放比例估算，旧配置在那一轮里抽到说明录音的不到 1 条，所以它不是那一轮纠偏变差的原因。
- **计划**：交付版的五份（热词、无语音、漏识、TS 缺席、TS 在场）在起点上重挖，再加一份 `ctx`。
- **`ctx` 的挖题候选**：
  - 合成说字数据 `train`、`train_ts`，含成对的保持题；
  - 配方里的说明录音子集：emilia2_mix `train_2spk`/`train_3spk`、emilia2_dialog `train_2spk`、wenetspeech_clean `train`、emilia2_short `train_zh`。
  - 这些子集和完整来源共用来源键。`grpo_mining` 和 `GRPOPromptPlan` 遇到重复的键都取最后出现的那个，而子集排在完整来源之后，所以按键挖到的就是子集，两边一致。
- **奖励**：交付版的四项（准确率、出字判断、热词、格式），加 `qwen3_asr_context`。权重训练前定，先按和热词相同的 1.0 考虑。
- **评测**：上下文纠偏 v2（think 关、开）、守门 8 项、场景 TS。判定在训练前写定。

## 已完成

- `tasks/00-context-targets`：目标表，见该任务 README。
- `scripts/mining_plans.py`：从 08 的 `mining_plans_n16.py` 复制，加了 `ctx` 计划：
  - 先看出字与否，有分歧就归入原来的漏识 / TS 在场；
  - 出字没有分歧时，看 16 次采样里“每个目标都写对”的少数一方是否 ≥ 2 次，是则归入 `ctx`。
  - 用手工样例核对过：14 对 2 错归入 `ctx`；15 对 1 错不留；没有目标的记录不归入 `ctx`；出字有分歧时优先归入漏识。

## 待做（第 4 问之后）

- 定起点和 SFT 配方，生成 GRPO 数据配置：回放、六份计划、`grpo.context_targets`。如果配方变了，要重跑目标表。
- 挖题脚本：交付版的 7 条普通识别线和 7 条 TS 线，加一条 `ctx` 线。普通识别服务挖合成 `train` 和说明录音子集，TS 服务挖合成 `train_ts`。
- 训练配置：交付版第十轮的参数，奖励加 `qwen3_asr_context`，开联合 MTP（GRPO lr 1e-5）。
