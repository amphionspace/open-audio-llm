# 上下文纠偏的 GRPO 计划（2026-11-20 起）

**状态**：准备中，还没有训练。2026-11-21 起点定为上下文纠偏实验第 7、8 问选出的 J（`template-synth-ts-twins` 的 checkpoint-3000）。配置已按 J 重新生成，目标表定稿为 `context_correction_clean_zh@grpo-targets-v2-20261121`（audio-data-contract #284 已合入，registry 已发布，COS 已镜像）。判定已在训练前写定，见“判定”。

**要回答的问题**：在 GRPO 里加一份上下文纠偏计划，能不能让 think 关时真实录音上的显式说明题多改对，同时守门和场景 TS 都不变差？
- Emilia 显式说明题现在 think 关只有 44%。
- 主要失败是听到了说明，却没有把它关联到前面的目标字。这是行为问题，奖励可以直接优化。

**依据**：
- 上下文纠偏实验 `runs/zh-ctx-clue-target-20261117`（分支 `feat/ctx-clue-target-think`）README 的“真实录音失败归因”和“下一步：GRPO 上下文纠偏计划”。
- 调研记录 `docs/research/2026-11-20-context-correction-related-work.md`。外部的主流做法是 SFT 之后做 RL，奖励给关键词加权，并惩罚多改。
- 奖励 `qwen3_asr_context` 和数据配置项 `grpo.context_targets` 已合入 main（#170），说明见 `docs/guides/training.md` 的 Qwen3-ASR GRPO。

## 设计

- **起点**：修好场景 TS 的上下文纠偏 SFT checkpoint。第 4 问查明退化来自合成说字数据，第 5 问说明整份去掉的代价太大，第 6 问两种改法都不通过。
  - 第 7、8 问选定 J `template-synth-ts-twins`：第二轮配方加合成音色 TS 场景（30% 孪生缺席题），ae 模型缓存 `checkpoint-3000-061b86c3902e`。
  - J 场景 TS CER 3.0、误出字 16 条，上下文纠偏和真人语音 TS 都不比 template 差。
  - 要盯的项：上下文纠偏 v2 的 TS 题输出空 134 条（template 70），跟错人集误触发 +0.35（刚好不显著）。
- **回放**：用起点那轮 SFT 的配方，版本换成训练文本终版（`text-final-v1-20261120`）。终版包含说字清洗的修正，另有字母连写、英文写回等修正；主线也在切到终版。
  - 交付版那一轮 GRPO 的数据配置沿用了更早的配方，用的是未清洗的版本，也没有合成说字数据。本实验不沿用它。
  - 不沿用的原因：回放的参考答案必须和起点学过的写法一致，否则准确率奖励会把改对的字往回拉。按回放比例估算，旧配置在那一轮里抽到说明录音的不到 1 条，所以它不是那一轮纠偏变差的原因。
- **计划**：交付版的五份（热词、无语音、漏识、TS 缺席、TS 在场）在起点上重挖，再加一份 `ctx`。
- **`ctx` 的挖题候选**：
  - 合成说字数据 `train`、`train_ts`，含成对的保持题；
  - 配方里的说明录音子集：emilia2_mix `train_2spk`/`train_3spk`、emilia2_dialog `train_2spk`、wenetspeech_clean `train`、emilia2_short `train_zh`。
  - 这些子集和完整来源共用来源键。`grpo_mining` 和 `GRPOPromptPlan` 遇到重复的键都取最后出现的那个，而子集排在完整来源之后，所以按键挖到的就是子集，两边一致。
- **奖励**：交付版的四项（准确率、出字判断、热词、格式），加 `qwen3_asr_context`。权重训练前定，先按和热词相同的 1.0 考虑。
- **候选**：同 08，第 200/300/400 步的平均（avg3），和它与起点等权的插值（插值 50%）。
- **评测**：上下文纠偏 v2（think 关、开）、守门 8 项、跟错人、场景 TS，都 think 关（上下文纠偏另测 think 开），ae 0.14。
  - 起点 J 在同一个节点上、用同一份代码（`shared/code-v1`，main f044443）全部重测，和候选同口径。第 7、8 问的 J 用的是上下文纠偏实验的代码（main 0a1b117），只作参照。

## 判定（训练前写定，2026-11-21）

avg3 和插值 50% 各自和起点 J 配对比较，think 关。

1. **显式说明题改对更多**（主判定）：Emilia“需要纠正 · 显式说明”的 516 条（`ctx_v2_mined_groups.py` 的分组；第 7 问 J think 关 45.2%，template 44.2%），新对多于新错，精确符号检验双侧 p < 0.05。
2. **纠偏不变差**：上下文纠偏 v2 的全部保持率、Emilia 全部准确率，CI 下限都不低于 −1 个点（同 SFT 各问）。
3. **TS 不变差**：场景 TS 的 CER、误出字率；中文 TS-ASR 测试集 v12 和跟错人 first-other v3 的 CER、误触发率、漏识别率。每项都不显著更差（候选 − 起点的 95% CI 下限 ≤ 0）。
4. **守门不变差**：08 守门表里的 asrreg 中文 CER、术语命中率、Common Voice 中文 CER、asrreg 英文 WER、GigaSpeech 无语音误触发率、Common Voice 英文 WER、警务全量 CER 和术语命中率，每项都不显著更差（术语命中率看 CI 上限 ≥ 0）。
   - 鼎桥客户两项、两项热词和 CHiME-4 样本少或不在 `compare` 范围内，同 08 只作参考。
- **只记录**：上下文纠偏 v2 的 TS 题输出空的条数（J 134），场景 TS 目标在场输出空的条数（J 6/1,830），think 开的上下文纠偏，各项对交付版的差。
- **结论**：
  - 四项都满足：该候选进入交付评估（对交付版的各项差由用户定夺，同 08）。avg3 和插值都满足时取 avg3，因为插值等于往起点退一半，GRPO 学到的也减半。
  - 只有主判定不满足：说明这个份额和奖励下 GRPO 没把显式说明题改对，按数字记录，按失败类型分析，不定下一步的参数扫描。
  - 第 2–4 项有不满足：按数字记录，按来源定位，不进交付评估。

## 已完成

- `tasks/00-context-targets`：目标表，登记为 `context_correction_clean_zh@grpo-targets-v1-20261120`（audio-data-contract #279），见该任务 README。
- `scripts/mining_plans.py`：从 08 的 `mining_plans_n16.py` 复制，加了 `ctx` 计划：
  - 先看出字与否，有分歧就归入原来的漏识 / TS 在场；
  - 出字没有分歧时，看 16 次采样里“每个目标都写对”的少数一方是否 ≥ 2 次，是则归入 `ctx`。
  - 用手工样例核对过：14 对 2 错归入 `ctx`；15 对 1 错不留；没有目标的记录不归入 `ctx`；出字有分歧时优先归入漏识。

## 已完成（2026-11-21）

- **代码快照** `shared/code-v1`：main f044443，含 `qwen3_asr_context` 和 `grpo.context_targets`。
- **`scripts/final_versions.py`**：把配置里的版本映射到训练文本终版。先找“原版本 + `-final-v1-20261120`”，没有再看 `superseded_by`；这两条规则文本终版那边已确认。template 配方 75 个版本中 74 个有终版，剩下的是合成说字数据，它没有终版。
- **`scripts/make_configs.py ARM START TARGETS`**：生成 GRPO 数据配置、训练配置、14 路挖题服务配置、挖题来源列表和 `experiment.yaml`。
  - 训练配置同 08（第十轮）：lr 1e-6、beta 0.04、400 步；奖励加 `qwen3_asr_context`（权重 1.0）；开联合 MTP（lr 1e-5，从起点的 `mtp/` 接上）。
  - 数据配置：`ctx` 份额 0.08，其余五份计划的份额同 08。
  - 挖题来源都核对过在数据配置里，ctx 线是合成数据加说明录音子集。
  - 带版本号的记录 ID 名单要改写前缀。只有 wenetspeech_clean 的说明录音子集名单是这种，它的记录由 Lhotse cut 构建，ID 里带版本号，按旧版本写的名单在终版上一条都匹配不上。改写后的名单随目标表新版本一起登记。
- **`scripts/finalize_targets.py`**：把目标表的来源键和记录 ID 换成终版，直接读登记的 Lance 记录索引或 JSONL，按终版转写核对字符位置。
  - 对不上的目标就近移位；每行都用奖励函数本身核对，得分为 1.0。
  - 用 v1 目标表试跑：7,011 行全部找到，9,349 个目标中 3 个移位，没有丢失，用时约 2 分钟。
- **`scripts/mining.sh ARM TARGETS`**：08 的挖题线，加 4 条 ctx 线（普通识别、TS 各 2 条，每个候选每条线挖一次）。之后由 `scripts/mining_plans.py` 分成六份计划。

## 已完成（2026-11-21，起点定为 J 之后）

- **目标表**：`build_context_targets.py` 只读挖题用的合成 `train`、`train_ts`；J 配置里的孪生 split 只进回放，孪生题没有纠偏标注。
  - 在 template 配置上重跑，输出和 v1 逐字节相同；在 J 配置上，输出和 template 的逐字节相同。
  - `finalize_targets.py` 定稿：7,011 行全部找到，9,349 个目标中 3 个就近移位，没有丢失。
  - 登记为 `context_correction_clean_zh@grpo-targets-v2-20261121`（audio-data-contract #284，2026-11-21 合入，registry 已发布）。内容是目标表、`summary.json`、`state.json` 和改写后的 wenetspeech_clean 名单（923 条）。
  - COS 镜像：`amphion-audio-backup-1373259071:audio-data-contract/datasets/context_correction_clean_zh/versions/grpo-targets-v2-20261121/`，4 个文件、1,824,186 字节，打包为 1 个分片，已核对。
- **配置**：`make_configs.py template-synth-ts-twins <J> <v2 targets>` 重新生成，dry-run 通过。孪生来源在回放里，带在线混音；挖题只取合成 `train`、`train_ts`。
- **评测脚本**：`make_eval_configs.py`（抄自上下文纠偏实验）、`make_gate_configs.py`（抄自 08，加跟错人和场景 TS）、`run_eval.sh`。
- **节点脚本** `scripts/acp-node.sh`：依次执行挖题 `mining.sh`、训练 `train.sh`、评测 `eval_after_training.sh`。
  - 按 08 的用时，挖题约 1 小时，训练约 40 分钟，评测约 2.5 小时。
  - 执行器目录 `/workspace/workspace/mingdong/scratch/acp-ctx-grpo`。

## 待做（第 6 问之后，已由上面完成，留作记录）

1. 用胜出组的 SFT 数据配置跑 `build_context_targets.py`，再跑 `make_configs.py`，然后用 `finalize_targets.py` 定稿。
2. 把目标表登记为 `context_correction_clean_zh@grpo-targets-v2-<日期>`，内容是目标表和改写后的名单，并镜像到 COS。
3. 训练前写定判定：上下文纠偏 v2（think 关、开）、守门 8 项、场景 TS，比较对象是起点和交付版。
4. 用一个 ACP 节点依次跑挖题、GRPO 400 步和评测。
