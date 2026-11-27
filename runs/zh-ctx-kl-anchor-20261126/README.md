# 上下文纠偏 SFT 加回放 KL：从 11.17 起训（2026-11-26）

**状态**：2026-11-27 后续 2（回放 KL 加在场判断项）进行中。11-26 两臂、后续 1（CE 只训正文）两臂的纠偏收益都成立，TS 都没保住，都不是替换候选；见“结果”“后续 1 结果”。

## 要回答的问题

从 11.17 直接续训上下文纠偏数据时，把“回放 11.17 原有任务”的交叉熵换成“对 11.17 的 KL”，能不能拿到真实录音上的纠偏收益，同时 TS 不回退？

## 依据

- **TS 回退的根因是续训本身**（`runs/zh-ctx-grpo-k-20261122` README“TS 回退的来源”）。
  - 在 GRPO 模型上用配方标注续训 SFT，会把模型拉回 SFT 状态，冲掉 GRPO 校准好的出字判断。
  - 不含任何纠偏数据的对照组也一样：从 11.06 续训，v12 误触发 2.70 → 3.49\*。
  - K 从 11.17 续训，同一机制：v12 漏识别 2.76 → 3.75\*。
- **真实录音的纠偏收益来自说字清洗的说明录音**（`runs/zh-ctx-clean-pair-20261125` README“结果”“后续结果”）。
  - 在 11.06 上，对照组比 noclean 在没见过的真实题上高 1.6\*。两者只差说字清洗：9 个数据版本的修正文本，加 5 个说明录音子集上采样，每条约见 3 次。
  - 这 5 个子集共 2,075 条：emilia2_mix 2 人 326、3 人 316，emilia2_dialog 2 人 229，wenetspeech_clean 923，emilia2_short 281。
- **差值合并不能用**：两次 3000 步训练只差约 1% 的数据，增量几乎正交（余弦 0.155），差值大部分是训练随机性。pm-clean 的 TS 回退就来自这里。
- **终版文本已经包含说字清洗**：9 个说字清洗版本都是 text-final-v1/v2 的起点（`catalog/text_final_v1.yaml` 的 `derived_from`）。
  - 11-21 起所有新训练都用终版文本，所以主线下一轮 SFT 本来就带说字清洗。
  - 11-25 留给用户的“主线 SFT 是否换用说字清洗版本”不再需要单独决定。
  - 上采样那一份不在终版里，属于配方。
- **回放 KL 已有可用的做法**：TS 线任务 25/28（`oal-clean-teacher` `runs/ts-overlap-20261119`，`scripts/ft_enc_replay.py`）。
  - 从 11.17 起训，回放 11.17 原有任务时只算对 11.17 的 KL，不算交叉熵。开训时 KL 为 0，只起拉住漂移的作用。
  - 那边只训音频塔，LLM 冻结；本实验 LLM 也要训，因为纠偏收益在 LLM 和音频塔里都有（K README“拆分诊断 · 结果”）。

## 设计

- **起点和老师**：11.17（ae 模型缓存 `model-afea989da22b77adc0ce9365`，与交付包权重逐字节相同）。老师是一份冻结的完整 11.17。
- **可训练部分**：与 spelling-think 对照组相同，冻结词嵌入（和 lm_head 共享）和音频塔的卷积前端，其余全训。
  - MTP 不训练；导出的 checkpoint 链接 11.17 的 `mtp/`，它只影响接受长度，不影响贪心输出。
- **损失**：每步每卡一个 CE 样本（纠偏流）加一个回放 batch；损失 = CE + 1.0 × KL(11.17 ‖ 学生)。
  - KL 覆盖回放的全部监督 token，KL 权重 1.0 同 TS 线任务 25/28。
- **CE 流，两臂只差这一项**（`scripts/make_configs.py` 生成）：
  - **臂 A `real`**（`configs/ce-data-real.yaml`）：上面 5 个说明录音子集，版本和 include 列表换到最新终版 v2（wenetspeech_clean 的列表带版本前缀，已改写到 `configs/records/`）。
  - **臂 B `real-synth`**（`configs/ce-data-real-synth.yaml`）：再加合成说字数据 `context_correction_synth_zh@train-v1-20261114` 的 `train`，2,984 条普通识别。
    - 依据：去掉合成数据后，think 关的 Emilia −1.1\*（第 5 问）。合成数据主要让显式说明题多改对（J 显式 +44 / 隐式 +2）。
    - 不加 `train_ts`：它让模型把“是 TTS 音色”当成目标人线索（场景 TS 误报 21.4 → 55.7\*）。
    - 不加孪生场景：长对话漏人（154/289）。
    - 不加隐式题数据：第 3 问没过，已撤出配方。
  - 每个来源的权重等于记录数，一个 epoch 每条约抽到一次。
- **回放流**（`configs/replay-data.yaml`）：TS 线任务 25/28 的回放配置。
  - 125 个来源，已在 16 卡上预检通过。
  - 覆盖中英西葡普通识别、热词、中英 TS 在线混音、真实 TS 录音和各类负样本，都是终版文本。
  - fleurs_zh 换到终版 v3。
- **步数和学习率**：
  - 步数 = 3 × 每卡一个 epoch 的 CE 批数，即每条 CE 记录约见 3 次，同第二轮配方的规则。
  - 4 卡时，臂 A 约 1,557 步，臂 B 约 3,795 步。
  - 学习率峰值 3.55e-6，前 10% 线性 warmup，之后 cosine 降到 1e-6，梯度裁剪 1.0，同对照组。
  - weight decay 取 0：起点是交付模型，往 0 衰减是 KL 拉不住的漂移。
- **checkpoint**：一半处和最后各存一个完整模型目录，在 `/workspace/data/ae-runs/ctx-kl-anchor-20261126/<臂>/`。判定只看最后一个，一半处的只留作剂量诊断。
- **训练脚本**：`scripts/ctx_kl_train.py`，经实验启动器运行（`experiment.yaml`，任务 `train-real`、`train-real-synth`）。
  - 每 10 步写一行 `logging.jsonl`（loss、ce_asr、replay_kl 等），由启动器上传 W&B。
  - 另写各任务类别的回放 KL：`replay-kl-by-task.jsonl`。

## 评测

同一节点，训练完以后（`scripts/acp-node.sh`）。两个候选的模型名是 `klreal`、`klsynth`。
- **纠偏 v2**（think 关）：两臂都测，用 `runs/zh-ctx-grpo-k-20261122` 的配置生成和运行脚本，和 11.17 已有的 run 同一套配置。
- **交付评测 27 组**：两臂都测，用 `runs/zh-ctx-grpo-20261120/scripts/acp-delivery-one.sh`，和 11.17 及之前各次诊断同一套配置。
- 推理栈与之前相同：vLLM 服务，MTP 草稿用 11.17 的。

## 判定（候选对 11.17，配对 bootstrap 95% CI）

1. **TS 保住**：以下各项都不显著变差，口径同之前各次合并诊断。
   - v12、跟错人、英文 TS 的误触发率和漏识别率；
   - 西语 TS v1/v2、跨语种（西语）的误触发率。
2. **真实录音纠偏收益**：纠偏 v2（think 关）上，目标词训练里没出现过的真实题（1,680 条），准确率显著好于 11.17（CI 下限 > 0；11.17 为 67.20）。
3. **交付评测**：27 组里列出所有显著变差的项。按交付评估的标准，每项都不显著差于 11.17 才能作为替换候选；是否替换由用户决定。

## 分支

- **1、2 都成立**：这一臂是候选，看第 3 条。
  - 两臂都成立时，取第 3 条变差项少的；一样多时，取没见过的真实题提高更多的。
  - 交付评测全表交给用户。
- **1 不成立（KL 没拉住 TS）**：
  - 看 `replay-kl-by-task.jsonl` 里哪类任务的 KL 涨得最多；
  - 看 CE 流里 871 条 TS 正样本（emilia2_mix、emilia2_dialog 的说明录音）是不是把 TS 往出字推。
- **2 不成立（收益没出来）**：
  - 看显式 / 隐式分组（`ctx_flip_breakdown.py`），和对照组在 11.06 上的分组比；
  - 看一半处的 checkpoint，判断剂量方向。
- **只记录、不判定**：
  - 长对话漏人、西语 WER、英文无语音误触发、场景 TS、ctx-v1；
  - 纠偏 v2 的其他数字；
  - 一半处 checkpoint、MTP 接受长度。

## 风险

- **CE 流里有 TS 正样本**：871 条说明录音是目标人在场的 TS 样本，单独算 CE 可能让模型更倾向出字。回放里的不在场样本由 KL 拉住，判定 1 检验这一点。
- **回放没有中英混说数据**：TS 线任务 25/28 的中文普通识别漂移约 74% 在 TALCS 中英混说上，而回放里没有这类数据，KL 拉不住。本实验 LLM 也训练，这类输入上的漂移风险更大。判定 3 的中文普通识别各项会显示出来。
- **MTP 没训练**：LLM 变了，草稿接受长度可能下降，只影响速度，不影响输出。
- **说明录音也在回放的完整来源里**：被抽到时 KL 会把它拉回 11.17 的写法。按回放比例估算，整个训练抽到的不到 10 条，可以忽略。

## 执行

- 2026-11-26：配置 dry-run 通过（两臂的命令、GPU 0-3 / 4-7、端口、W&B 转发 10.119.18.89:17920、两项数据预检）。
- 开发机 IP 从 10.119.17.252 换成 10.119.18.89，`runs/zh-ctx-grpo-20261120/scripts/acp-delivery-one.sh` 的 W&B 转发地址已改为新 IP。
- 2026-11-26 15:55Z：提交 ACP pt-53size8p（1 节点 8 卡），节点上的 dev_runner 执行 `scripts/acp-node.sh`。
- 17:23Z：臂 A attempt 001 在第 1 步崩溃，原因是老师模型的复制方式。
  - 老师原来是 `copy.deepcopy(model)`。`attach_sep_token` 把加载出来的 thinker 原有的 `forward`、`get_audio_features` 存在闭包里，deepcopy 不复制闭包，老师的前向实际调用学生的 thinker，又读不到 `enroll_n_frames`，TS 样本少插 sep token，`masked_scatter` 断言失败。
  - 就算没有 TS 样本，这个“老师”也等于学生，KL 恒为 0，等于没有锚定。
  - 修复：老师单独再加载一份 11.17。臂 B attempt 001 启动在修复之前，在数据预检阶段中断。
- 17:27Z：同一节点上两臂都以 attempt 002 重跑，之后接着评测（节点脚本 `20-klanchor-retry`）。
- 18:09Z 臂 A、19:00Z 臂 B 训练完成，W&B 远端都是 finished（数字见 `tasks/01`、`tasks/02`）。
- 19:01Z：评测阶段给臂 A 选成了一半处的 `checkpoint-778`。
  - 原因：`acp-node.sh` 的 `final_ckpt` 用 `sort -t- -k2 -n`，但路径里的 `ctx-kl-anchor-20261126` 自己带连字符，第 2 个字段不是步数，结果按字典序排。臂 B 碰巧选对。
  - 19:02Z 全部停下（节点脚本 `21-stop-eval`）：还没有评测进程启动，只留下 4 个服务端的 attempt 001 记录（interrupted，生效配置里是 checkpoint-778），没有评测产物，没有 W&B run。
  - 修复：改用 `sort -V`，并要求选中的 checkpoint 的 `SOURCE.json` 满足 `step == steps`。
- 19:03Z：只重跑评测阶段（节点脚本 `30-klanchor-eval`），klreal = `real/checkpoint-1557`，klsynth = `real-synth/checkpoint-3795`。
- 21:54Z：评测全部完成，退出码 0，节点由 `stop_when_done.sh` 停掉。

## 结果（2026-11-26）

| | klreal | klsynth |
|---|---|---|
| 判定 1：TS 保住 | 不成立：v12 FA 2.46 → 2.72\*，跟错人 FA 2.58 → 2.96\*，英文 TS miss 2.94 → 3.28\* | 不成立：6 项 FA 全部显著变差（v12 3.65\*，英文 TS 8.41\*） |
| 判定 2：没见过的真实题（11.17 为 67.20） | 70.60\* [+2.08, +4.77] | 72.92\* [+4.29, +7.26] |
| 判定 3：交付 27 组 | 12 组有显著变差 | 14 组有显著变差 |

- 两臂都不是替换候选。
- TS 回退是“在场判断”token 的偏移：CE 流的每条样本都在训练 `language <语种>`，没有 `None`；回放 KL 对全部 token 取平均，拉不住临界样本。
- 数字、逐条翻转和诊断见 `tasks/03-eval/README.md`。

## 后续：CE 只训正文（2026-11-26 加，训练前写定）

**要回答的问题**：CE 流不训“在场判断”token，只训正文，TS 能不能保住，纠偏收益还在不在？

**依据**（`tasks/03-eval/README.md`“TS 回退的诊断”）：
- 两臂的 TS 回退都是 `language` 后面那个 token（`None` 还是语种）在动，正文没有问题。
- CE 流的每条样本标签都以 `language Chinese<asr_text>` 开头（开发机 CPU 抽查 60 条，全是这个前缀，`<asr_text>` 各出现一次；`/workspace/workspace/mingdong/scratch/ctx-kl-anchor/label_prefix_check.py`），没有 `None`。所以 CE 一直在直接训练这个判断 token。
- 纠偏收益在正文 token 上，在场判断本来就应该由 11.17 和回放 KL 决定。

**设计**：只改一处，其余和 11-26 的两臂完全相同（数据、步数、学习率、回放 KL、起点、种子）。
- `scripts/ctx_kl_train.py` 的参数 `ce_text_only: true`：CE 标签里第一个 `<asr_text>` 及之前的部分（`language Chinese<asr_text>`）设为 -100，不算交叉熵；每条 CE 样本都必须有 `<asr_text>`，否则报错。
- 两臂：`configs/train-real-textce.yaml`（任务 04，模型名 `kltreal`）、`configs/train-real-synth-textce.yaml`（任务 05，模型名 `kltsynth`）。
- 评测同 11-26（`scripts/acp-node.sh textce`），任务 06。

**判定**：同上面“判定”1–3，候选对 11.17。另外和 11-26 同臂比较，只记录。

**分支**：
- **1、2 都成立**：看第 3 条，规则同上；交付全表交给用户。
- **1 不成立**：在场判断是经由共享表示偏的，不是 CE 直接训判断 token。用同样的逐条翻转看偏的方向，再考虑让回放 KL 在判断 token 上不被稀释（改设计，不扫权重）。
- **2 不成立**：纠偏收益有一部分和判断 token 绑在一起，看显式 / 隐式分组和 11-26 同臂的差。
- **预计不变、只记录**：11-26 两臂的普通识别小幅回退（babble3/4、西语识别、asr-benchmark、police 的 CER/WER）出在正文上，这次的改动管不到；判定 3 会照样列出来。

**执行**：
- 2026-11-26 22:13Z 提交 ACP pt-mweqkx4c（1 节点 8 卡，节点 10.119.18.23；A 线、TS 线确认不冲突），执行器 `/workspace/workspace/mingdong/scratch/acp-ctx-klanchor-textce`，节点脚本 `acp-node.sh textce`，评测完成后由 `stop_when_done.sh` 停节点。
- 22:14Z 臂 A（任务 04）启动，22:23Z 臂 B（任务 05）启动；两臂 W&B 远端都在更新。
- 屏蔽生效的证据：第 1 步和 11-26 臂 A 是同一个 batch（同一种子），CE 从 0.1285 变成 0.1708。去掉的是容易预测的前缀 token，正文按 token 平均自然更高。
- 2026-11-26 23:57Z 两臂训练完成（W&B 远端 finished，数字见 `tasks/04`、`tasks/05`），接着评测；11-27 03:08Z 评测全部完成，退出码 0，03:09Z 节点停掉。

## 后续 1 结果（2026-11-27）

| | kltreal | kltsynth |
|---|---|---|
| 判定 1：TS 保住 | 不成立：v12 FA 2.65\*、v12 miss 2.93\*、英文 TS miss 3.41\* | 不成立：v12 FA 2.91\*、跟错人 FA 4.30\*、西语 TS v1/v2 FA 3.72\* / 3.58\* |
| 判定 2：没见过的真实题（11.17 为 67.20） | 70.00\* [+1.58, +4.10] | 72.74\* [+4.03, +7.15] |
| 判定 3：交付 27 组 | 15 组有显著变差 | 13 组有显著变差 |

- 两臂都不是替换候选。
- 去掉 CE 对在场判断 token 的训练，合成数据那一臂往“出字”的偏移少了约一半（v12 不在场新出字 187 → 112，英文 TS 256 → 103），纠偏收益和 11-26 同臂没有显著差别。剩下的偏移经由共享参数传过来，走写定的分支“1 不成立”。
- 只记录：kltsynth 长对话 scene-asr CER 4.72 → 8.17\*，是长段口述提前结束（结束 token 也是 CE 流在训的一个“决定”）；纠偏 v2 / ctx-v1 的合成 TS 题上 kltsynth 整句不出字变多。
- 数字、逐条翻转和诊断见 `tasks/06-eval-textce/README.md`。

## 后续 2：回放 KL 加在场判断项（2026-11-27 加，训练前写定）

**要回答的问题**：回放 KL 在在场判断 token 上单独加一项、不再被正文 token 稀释，TS 能不能保住，纠偏收益还在不在？

**依据**（后续 1 的结果，加上对训练脚本和回放标签的核查）：
- 后续 1 去掉了 CE 对判断 token 的直接训练，剩下的偏移经由共享参数传过来。能直接拉住这个 token 的只有回放 KL。
- 回放 KL 是每个样本内对监督 token 取平均（`scripts/ctx_kl_train.py` `replay_kl`，每步每卡 1 个回放样本）。开发机 CPU 抽查回放流 400 条（`/workspace/workspace/mingdong/scratch/ctx-kl-anchor/replay_label_check.py`，日志 `replay-label.log`）：
  - 400 条的监督文本都以 `language <X><asr_text>` 开头，`None`、各语种名都是单个 token，判断 token 总在第 2 个监督位置；
  - TS 正样本占 45.8%，监督 token 中位数 27，判断 token 只占 1/27；TS 负样本占 13.5%，固定 4 个 token（`language None<asr_text><|im_end|>`），占 1/4；
  - 合起来，判断 token 在回放 KL 里约占 5% 的权重。
- 锚定看得到偏移：TS 负样本除判断 token 外的 3 个 token 是确定的，所以它的 token 平均 KL × 4 约等于判断 token 的 KL。后续 1 两臂最后 10% 约 0.02（0.0052 × 4、0.0053 × 4），开训第 1 步约 1e-4 以下。问题在于给它的权重太小。

**设计**：只改一处，其余和后续 1 的两臂完全相同（CE 只训正文、数据、步数、学习率、回放 KL、起点、种子）。
- `scripts/ctx_kl_train.py` 的参数 `decision_kl_weight`：回放损失 = token 平均 KL + `decision_kl_weight` × 判断 token 的 KL。判断 token 是每个回放样本第 2 个监督 token；第 1 个必须是 `language`，否则报错。
- 取 1.0：每个回放样本的判断 token 和这个样本整句的 token 平均 KL 同等分量。不扫这个值。
- 每 10 步另记 `replay_kl_decision`（W&B）和各类回放任务的判断 token KL（`replay-kl-by-task.jsonl` 里的 `<任务>_decision`）。
- 两臂：`configs/train-real-deckl.yaml`（任务 07，模型名 `kldreal`）、`configs/train-real-synth-deckl.yaml`（任务 08，模型名 `kldsynth`）。评测同 11-26（`scripts/acp-node.sh deckl`），任务 09。

**生效核查**：
- 第 1–3 步的日志里，判断 token KL 和 token 平均 KL 一样只有 fp32 / bf16 的数值差（约 1e-4 量级），但不为 0（恒为 0 说明取错了位置）。
- 最后 10% 的 TS 负样本判断 token KL 明显低于后续 1 的约 0.02。

**判定**：同“判定”1–3，候选对 11.17。另外和后续 1 同臂比较，只记录。

**分支**：
- **1、2 都成立**：看第 3 条，规则同上；交付全表交给用户。
- **1 不成立，回放上的判断 token KL 已经明显降低**：回放分布上拉住了，评测上的翻转发生在回放没覆盖到的临界样本（1 秒注册音、注册音被污染、相似音色）。下一步是回放覆盖的问题，整理证据交给用户，不再加训练。
- **1 不成立，判断 token KL 没有降低**：查这一项为什么没起作用（实现、梯度量级），不重跑。
- **2 不成立**：拉住判断 token 和纠偏收益有冲突，看显式 / 隐式分组和后续 1 同臂的差。
- **预计不变、只记录**：普通识别的小幅回退（babble3/4、西语识别、asr-benchmark、police）出在正文上；kltsynth 的长段提前结束出在结束 token 上。这次的改动都管不到，判定 3 会照样列出来。

**执行**：待启动。

<!-- execution-status -->

执行状态：completed；质量状态：未核实。

开始：未知；结束：未结束；退出码：未知。

<!-- /execution-status -->
