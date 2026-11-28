# 上下文纠偏放进主线一轮的成对对照（2026-11-27，训练前写定）

**状态**：完成。K0 一轮 11-27 20:42Z、第二个种子的 K0（k0s2）11-28 04:04Z 跑完。两个基线上判定都落在“1 成立但 2 有回退”（见“结果”“第二个种子的结果”）。按用户 11-28 的安排，这一包不单独上线，取舍写进“并入主线交接单”。

## 要回答的问题

上下文纠偏的数据和方法放进主线的一轮训练（先 SFT 再 GRPO）以后，收益有多少？会不会让其他能力回退？

之前的结论都来自“从 11.17 续训”，回退里混着“在 GRPO 模型上续训”本身的代价，分不开（`runs/zh-ctx-grpo-k-20261122` README“TS 回退的来源”）。这里用成对对照把这部分代价抵掉：两边都从 11.17 走同样的 SFT → GRPO，只差纠偏这一包。

## 依据

- 已有带纠偏的一轮：K（`runs/zh-ctx-clue-target-20261117` 第 9 问，从 11.17 起 SFT 3000 步）加上 K 的 GRPO（`runs/zh-ctx-grpo-k-20261122`，400 步，avg3，模型名 `kgrpo`）。交付评测和纠偏 v2 都已经跑过。
- 缺少的是同一轮里不带纠偏的那一边。本实验只补这一边：K0。

## 设计

K0 = K 那一轮去掉纠偏部分，其余完全照 K 的实际配置（`scripts/make_configs.py`，从 K 和 kgrpo 生成好的配置文件改，不重新生成版本）。

- **去掉的（纠偏这一包，按 K 实际训练的内容）**：
  1. 合成说字数据：`context_correction_synth_zh` 的 train、train_ts、train_ts_twins 三个来源；
  2. SFT 的 think 段（第二轮 think：说明录音和合成数据上的 think 行，加上很小比例的空 think 和普通 think），所以 K0 的每条样本都按普通识别 / TS 识别训练；
  3. GRPO 的 ctx 计划、context_targets、`qwen3_asr_context` 奖励和两路 ctx 挖题。
- **保留的**：
  - `context_correction_synth_zh` 的 train_asr_dialog。它是 K 为修长对话漏人加的普通识别长对话，不是纠偏数据。
  - 其余 131 个 SFT 来源（GRPO 133 个）的版本和权重。
  - 西语回放、热词设置、训练参数：起点 11.17、3000 步、学习率 3.55e-6、联合 MTP、种子。
  - GRPO 配方：第十轮的五份计划和份额，ctx 原来的 0.08 份额回到回放；400 步，avg3 取第 200/300/400 步。
- **代码**：SFT 用 K 的 code-v1（`oal-clue-target` 那份快照），GRPO、挖题和评测用 kgrpo 的 code-v3，都和 K 那一轮相同。
- **评测**：
  - 都和 kgrpo 同一套：纠偏 v2（think 关、开），交付 27 组（`runs/zh-ctx-grpo-20261120/scripts/acp-delivery-one.sh`，模型名 `k0grpo`）；
  - 另测 K0 SFT 本身的纠偏 v2（只记录）；
  - ae 固定用 0.14.0-eedcf15。打分口径和 kgrpo、11.17 已有的 run 相同，不混用 amphion-eval 11-27 新出的打分版本 5。

## 判定（kgrpo 对 k0grpo，配对 bootstrap 95% CI，think 关）

1. **收益**：纠偏 v2 上“需要纠正 · 明确说明写法”的真实录音（Emilia 516 条，`ctx_v2_mined_groups.py` 的分组），kgrpo 显著高于 k0grpo（CI 下限 > 0）。
   - 另记录：播客同类、独立合成题的 correct 题、语义推断题、保持率、过度纠正率。
2. **回退**：交付 27 组里，列出 kgrpo 显著差于 k0grpo 的全部项和幅度。
   - 一项只有在显著且绝对差 > 0.5 个百分点时，才记为“纠偏带来的回退”。0.5 来自 11-25 的估计：两次独立 SFT 之间的差约 0.4。
   - TS 的误触发和漏识别如果一升一降、两者之和不变差，记为门槛移动，不算回退。理由是 SFT 层面 TS 门槛在两次训练之间会移动 1–1.7 个点（`runs/zh-ctx-clean-pair-20261125` 后续结果）。
3. **结论规则**：
   - 1 成立且 2 没有回退：纠偏这一包在主线一轮里有净收益，可以进主线；
   - 1 成立但 2 有回退：把收益和回退的全表交给用户，由用户取舍；
   - 1 不成立：在一轮 SFT → GRPO 里，纠偏这一包拿不到明确说明类的收益，不进主线。
- **只记录**：两边对 11.17 的全表；K 对 K0 在 SFT 层面的纠偏 v2。

## 风险

- **只有一对**：运行间波动会混进差值，判定 2 的 0.5 个点门槛和 TS 门槛移动规则就是为此设的。差值落在临界附近时，再跑一个不同种子的 K0 来估波动，写在结果里。
- **K 是 11-22 跑的**：代码快照、数据版本、配置都用 K 当时的。集群和节点不同，不影响确定性以外的结果。
- **“CE 只训正文”这一条方法不在内**：K 的 SFT 训练器（ms-swift）没有按来源屏蔽标签前缀的开关。这里测的是 K 实际训练的那一包。
- **K0 去掉 think 以后，think 开的数字没有意义**：think 开的结果只对 K 记录。

## 执行

- ACP pt-f4eiiqi0（1 节点 8 卡），11-27 14:01Z 提交，14:02Z 分到节点 10.119.16.252（W&B 转发已放行）。节点脚本 14:11Z 放入，K0 SFT 随即开始。执行器 `/workspace/workspace/mingdong/scratch/acp-ctx-mainline-pair`，节点脚本 `scripts/acp-node.sh`，跑完由 `stop_when_done.sh` 停节点。
- 预计：SFT 约 1.5 小时，挖题约 1 小时，GRPO 约 40 分钟，评测约 2.5 小时，合计约 6 小时。

## 结果（2026-11-27，详见 `tasks/04-eval/README.md`、`tasks/04-eval/gate.md`）

**判定 1 成立，判定 2 有回退 → 按规则把全表交给用户取舍。**

- **收益**：Emilia 需要纠正 · 明确说明（516）45.54 → 47.67，+2.13 [+0.19, +4.07]；其中训练没见过的 469 条 +2.35 [+0.21, +4.48]。合成题 correct 率 +13 到 +32 个点，过度纠正不显著。
- **回退**（纠偏这一包带来的，显著且 > 0.5）：西语 TS 误触发 v1/v2 4.4 → 9.3 / 4.3 → 9.0（出在 K 的 SFT，全部不在场场景一起翻倍）；chime4 WER 8.97 → 11.39（出在 K 的 GRPO，漏识别）；ts-hotwords-es CER +1.45；ts-en 误触发 +1.31；scene-ts CER +1.06；first-other CER +0.65；ts-hotwords CER +0.51。
- **同时变好**：ctx-v1 CER 6.50 → 1.31，crosslingual-en / es WER -2.22 / -4.48，asr-benchmark 英文误触发 -1.78。
- **新发现**：对 11.17 的真实录音纠偏收益大部分已经在 K0 里（没见过的明确说明：11.17 40.30 → K0 45.84 → K 48.19）。K0 保留了 finals 里的 ctxclean 改正文本和 5 个带说明录音子集的上采样（按普通识别训），去掉的合成数据 + think + GRPO 纠偏只多约 2.4 个点。
- **第二个种子**：收益 CI 下限只有 +0.19，按“风险”第一条加跑 k0s2（`scripts/make_configs.py` ARMS：训练种子 43、数据种子 20261128、挖题 epoch +100，其余同 K0；任务 05-sft-k0s2、06-mining-k0s2、07-grpo-k0s2、评测名 k0s2grpo）。节点 pt-bije92x2（A 线让出，`/workspace/workspace/mingdong/scratch/acp-v99e` 的 `ctx-k0s2.sh`），等 TS 线 t29 守门约 22:45Z 结束、8 张卡全空后启动，约 6 小时。kgrpo 对 k0s2grpo 用同一个 gate.py 再算一遍；两个 K0 之间的差就是运行间波动。

## 第二个种子的结果（2026-11-28，详见 `tasks/08-eval-k0s2/README.md`）

- **收益扛住了波动**：Emilia 需要纠正 · 明确说明，kgrpo 对 k0s2grpo +2.52 [+0.39, +4.65]，判定 1 仍成立；两个 K0 之间 -0.39 [-2.13, +1.36]，不显著。纠偏这一包在真实录音上的净收益约 +2.3 个点。
- **稳定的回退**（两个基线上都显著且 > 0.5，K0 之间波动小）：西语 TS 误触发 v1 / v2 +4.7 到 +4.9；ts-hotwords-es CER +1.45 / +2.00；ts-en 误触发 +1.31 / +2.24；first-other CER +0.65 / +0.82；ts-hotwords CER +0.51 / +0.48（贴着门槛）。
- **撤回的回退**：chime4 WER（对第二个 K0 +1.10，不显著；两个 K0 之间就差 +1.32*）和 scene-ts CER（-0.32；K0 之间 +1.39*）和运行间波动一样大，不能算到这一包上。
- **门槛要按组给余量**：两个 K0 之间按同一条规则也会出现 scene-ts、chime4 两项“回退”，主线门槛对这两组要放宽到实测波动以上。
- **对 11.17 的收益大部分在 K0 里**：k0s2grpo 对 11.17 明确说明 +6.01*，和 k0grpo（+6.40*）一致。

## 并入主线交接单（2026-11-28）

**定位**（用户 11-28，经 open-audio-llm-71 转达）：上下文纠偏是单点研究，成果并入 71 的交付线。不单独出替换 11.17 的 checkpoint，kgrpo 不进交付。并入时机是任务 29 判定之后的下一轮主线（文本切到 final v4），门槛由 71 按本单加。

**推荐**：第 1–3 项进主线 SFT，第 5 项作为门槛；第 4 项（GRPO 纠偏奖励）标成“可选，代价未消除”。合成说字数据和 think 不进主线：它们是第 4 项那一包的主要代价来源，原因见“没解决的问题”。

| # | 项目 | 阶段 | 状态 |
|---|---|---|---|
| 1 | ctxclean 改正文本 | SFT | 已在 finals 里，不用做事 |
| 2 | 带说明录音上采样（+ 仅正文 CE） | SFT | 上采样推荐；仅正文 CE 在主线**未验证** |
| 3 | train_asr_dialog | SFT | 推荐，用 v4（竖线记录已隔离） |
| 4 | GRPO 纠偏奖励 + context targets | GRPO | 可选，代价未消除；targets 要在 v4 上重建 |
| 5 | ctx-v2 真实录音门槛集 | 评测 | 推荐 |

### 1. ctxclean 改正文本

- **数据版本**：9 个数据集的 `*-ctxclean-20261110` 版本（16 个条目）是 text-final-v1 的 `derived_from`，v2 / v3 / v4 一路继承。主线用 final v4 就已经包含，不用加配置，也不用改 trainer。
- **证据**：在 11.06 上成对比较（`runs/zh-ctx-clean-pair-20261125`），control 对 noclean，训练没见过的真实录音 +1.6*（新对比新错多 27 条：明确说明 +17，语义推断 +10）；noclean 对 11.06 只有 +0.1，不显著。
- **已知代价**：同一对里 TS 门槛有移动（v12 误触发 +1.25*、漏识别 -1.47*，两者之和 6.75 → 6.54，CER 14.51 → 13.32 变好）。两次 SFT 之间门槛本身就会移动 1–1.7 个点，单对分不开。
- **门槛**：第 5 项。

### 2. 带说明录音上采样（+ 仅正文 CE）

- **数据版本**：`context_correction_clean_zh@think-v2-20261113-final-v1-20261120` 的 5 个 include 列表，共 2,075 条，是说话人在录音里解释了写法的真实录音。

  | 来源 · split | 条数 | 权重（K0） |
  |---|---:|---:|
  | wenetspeech_clean · train（`task: asr`、`require_clean_pass: true`、`max_duration: 120`） | 923 | 0.7987 |
  | emilia2_mix · train_2spk | 326 | 0.2821 |
  | emilia2_mix · train_3spk | 316 | 0.2734 |
  | emilia2_short · train_zh | 281 | 0.2432 |
  | emilia2_dialog · train_2spk | 229 | 0.1982 |

  权重合计 1.7956，占 K0 SFT 总权重 192.19 的 0.93%。列表按 `split:record_id` 选；v4 只改文本，id 不变（抽查 emilia2_short：281 条全部在 `short-lance-20261023-final-v4-20261127` 里；11-28 PR #322 原地修正后又核了一遍，文件 sha256 和修正后的注册表一致，仍是 281 / 281）。这 4 个来源的 v4 都在 #322 原地修正的 44 个版本里，版本名不变、内容和 sha 变了，引用时重新读注册表，不用 #322 之前缓存的索引或哈希，并入时把 `version` 换成对应的 `-final-v4-20261127`，其余 4 个列表同样核对一遍命中数。
- **配置片段**（主线数据配置 `train:` 下加；版本换成主线当时的 v4 名）：

  ```yaml
  - dataset_id: wenetspeech_clean
    split: train
    version: train-punc-v1-20261102-final-v4-20261127
    include_records: /workspace/data/datasets/context_correction_clean_zh/versions/think-v2-20261113-final-v1-20261120/wenetspeech_clean/include-wenetspeech_clean_train-punc-v1-20261102-final-v1-20261120-train.txt
    task: asr
    require_clean_pass: true
    max_duration: 120
    weight: 0.7987
  # emilia2_mix train_2spk / train_3spk、emilia2_short train_zh、emilia2_dialog train_2spk 同样写法，
  # 列表路径和权重见 configs/train-data-k0.yaml 里带 include_records 的 5 个条目
  ```
- **trainer 改动**：K0 按普通识别训，标签前缀 `language Chinese<asr_text>` 也算 CE。“仅正文 CE”（屏蔽到第一个 `<asr_text>` 为止）只在 `runs/zh-ctx-kl-anchor-20261126` 的自写 trainer（`ce_text_only: true`）里测过：纠偏收益不变（kltreal 对 klreal 不显著），带合成数据那一臂“总要输出”的漂移减半（v12 不在场新输出 187 → 112）。但那是从 11.17 续训加 KL 锚定，不是主线配方，**主线上未验证**。主线 SFT trainer（ms-swift）没有按来源屏蔽标签前缀的开关，71 在主线实现。
- **证据**：K0 这一轮（第 1–3 项都在）对 11.17，训练没见过的明确说明 469 条：SFT 后 40.30 → 46.48*，GRPO 后 45.84*；没见过的全部 1,680 条 67.20 → 69.94*。三项各占多少，本实验分不开。
- **已知代价**：两个 K0 对 11.17 都有 TS 误触发上升，两次结果一致（k0grpo / k0s2grpo）：v12 +1.49 / +1.89，first-other +2.19 / +1.66，西语 v1 +1.19 / +1.33、v2 +1.30 / +1.24，crosslingual +0.80 / +1.59，scene-ts +10.0 / +7.1。同时漏识别和 CER 下降（v12 CER -0.27 / -0.41，西语 TS WER -1.0 到 -1.5）。k0s2grpo 另有 scene-ts CER +1.88、chime4 漏识别 +2.20，k0grpo 这两项不显著，属于波动大的组。这和“在 GRPO 模型 11.17 上再做一轮 SFT”的已知代价同一形态，本实验分不开它和第 1–3 项各占多少。主线从自己的 SFT 底座起训，预计没有这一层，但没验证过，并入后按第 5 项和交付门槛看 GRPO 之后的结果。

### 3. train_asr_dialog

- **数据版本**：`context_correction_synth_zh@asr-dialog-v1-20261122-final-v4-20261127`，split `train_asr_dialog`（1,841 条）。不要用 `train_asr_dialog-records-quarantine`（159 条目标里残留“反应|反映”这类竖线，TTS 把两个写法都念了）。K0 用的是旧版 `asr-dialog-v1-20261122`（2,000 条，含这 159 条）。
- **配置片段**：

  ```yaml
  - dataset_id: context_correction_synth_zh
    version: asr-dialog-v1-20261122-final-v4-20261127
    split: train_asr_dialog
    weight: 1.8274   # K0 的 0.95%
  ```
- **trainer 改动**：无，按普通识别训。
- **作用**：K 为修长对话漏人加的普通识别长对话（scene-asr 一人被整段漏掉：J 129 条 → K 4 条）。J 的漏人来自 ts-twins，而 twins 不进主线，所以它在主线里是否必要没有单独对照。k0grpo 对 11.17 scene-asr CER -2.49*。
- **已知代价**：旧版里的竖线被模型学会了（交付输出带竖线：k0grpo 126、kgrpo 54，11.17 0）；v4 已隔离。
- **门槛**：交付 27 组输出里，参考没有而输出带“|”的条数为 0（`scripts/gate.py` 的 `pipes()` 会数）。

### 4. GRPO 纠偏奖励 + context targets（可选，代价未消除）

- **代码**：`qwen3_asr_context` 奖励，open-audio-llm PR #170（11-20 已合入 main），kgrpo 用的 code-v3 = main 15eacbf。
- **配置片段**（kgrpo 实际用的，`runs/zh-ctx-grpo-k-20261122/configs/`）：

  ```yaml
  # train-grpo.yaml：--reward_funcs 加 qwen3_asr_context，--reward_weights 对应加 1.0
  # train-data-grpo.yaml 的 grpo:
    mined:
    - plan: .../plan-ctx.jsonl     # 挖题时多两路 ctx（带 context targets 的来源）
      share: 0.08                  # 从回放份额里拿出来
    context_targets:
    - /workspace/data/datasets/context_correction_clean_zh/versions/grpo-targets-v2-20261121/targets.jsonl
  ```
- **数据版本**：`grpo-targets-v2-20261121`，7,011 条。其中 5,901 条是合成说字数据（train 2,984、train_ts 2,917），约 1,110 条是带说明的真实录音（wenetspeech 320、emilia2_mix 404、emilia2_dialog 215、emilia2_short 143 等）。targets 按 final-v1 版本名和字符位置记录。v4 把数字换成阿拉伯数字后位置会变，**并入前要在 v4 上重建 targets**（新版本，不能沿用 v2；按 PR #322 修正后的文本建，真实录音来源的 v4 在 #322 里改过，合成数据的 4 个 v4 没改）。只用真实录音部分时，ctx 计划会小很多（kgrpo 的 plan-ctx 是 4,149 条）。
- **收益**：这一包（合成说字数据 + think + GRPO 纠偏）的净收益：Emilia 需要纠正 · 明确说明 +2.13 [+0.19, +4.07]（k0grpo 基线）、+2.52 [+0.39, +4.65]（k0s2grpo 基线），两个 K0 之间只差 -0.39（不显著），约 +2.3 个点。合成题 correct 率 +13 到 +33 个点。
- **已知代价**（同一包，对两个 K0 基线都显著且 > 0.5 的才列）：
  - 西语 TS 误触发 v1 / v2 约 4.4 → 9.3 / 4.3 → 9.0（+4.7 到 +4.9），出在 K 的 SFT（合成数据 + think），全部不在场场景一起翻倍；
  - ts-hotwords-es CER +1.45 / +2.00，ts-en 误触发 +1.31 / +2.24，first-other CER +0.65 / +0.82，ts-hotwords CER +0.51 / +0.48；
  - 原来记的 chime4 WER +2.42、scene-ts CER +1.06 和两个 K0 之间的波动（+1.32*、+1.39*）一样大，撤回；
  - 本实验没有“只加 GRPO 纠偏、不加合成 SFT 数据”的一臂，GRPO 纠偏单独的代价未测。西语 TS 误触发出在 SFT 段，所以只加 GRPO 纠偏、不加合成数据，有可能避开最大的一项，但没测过。
- **门槛**（如果加）：第 5 项的明确说明组，CI 下限 > 0（对不加它的同一轮）；交付 27 组不新增“显著且 > 0.5 个点”的回退，TS 误触发和漏识别按两者之和判；scene-ts 和 chime4 两次独立训练之间就差 1.4–1.9 个点，这两组的余量要放宽到 2 个点，或者用两个种子的基线。

### 5. ctx-v2 真实录音门槛集

- **评测集**：`asr/context-correction-zh@v2-20261117`，ae `ts-asr` evaluator，think 关，`split_group: evaluation`、`samples_per_source: 0`（全量），配置样例 `configs/eval-k0grpo-avg3/after-training-context-correction-v2.yaml`。
- **分组**（`scripts/gate.py` 的 `ctx_gate()`）：真实录音按 `context_correction_zh` 的 `mined_real.jsonl.gz`（Emilia 用 v2-20261116，播客用 v1-20261018），按 `metadata.corpus_transcript_correct` 分“需要纠正 / 原转写已对”，再按 category（explicit / implicit）分；训练是否见过按 `train_overlap` 拆。
- **基线数字**（ae 0.14.0-eedcf15，打分版本 4，think 关）：

  | 组 | 11.17 | K0 一轮（k0grpo） | 带纠偏一包（kgrpo） |
  |---|---:|---:|---:|
  | Emilia 需要纠正 · 明确说明（516） | 39.15 | 45.54 | 47.67 |
  | 其中没见过的（469） | 40.30 | 45.84 | 48.19 |
  | 没见过的全部（1,680） | 67.20 | 69.94 | 71.13 |
  | Emilia 原转写已对 · 明确说明（662，保持） | 88.07 | 90.94（+2.87*） | 91.84（+3.78*） |

- **建议门槛**：没见过的全部 1,680 条对 11.17 CI 下限 > 0；“原转写已对”的 662 条保持率不显著变差。
- **注意**：主线切到 final v4 后，模型输出阿拉伯数字。ctx-v2 的参考没有转换，打分版本 5 的数字转换两边不对称（v6 待用户定）。基线和候选必须用同一个打分版本重打，不能拿上表的数字直接比。

### 没解决的问题（研究继续，不阻塞并入）

- **合成说字数据让中文 TS 偏向“不在场”、西语 TS 偏向“总输出”**：在场判断的 token（`language` 后面那个）被整体推移，不是个别难例。KL 锚定和加在场判断项都没稳住（`runs/zh-ctx-kl-anchor-20261126`）。
- **长句提前结束**：带合成数据的臂 scene-asr CER 4.72 → 8.17*，长口述提前出 EOS。
- **多语种纠偏**（阿语、葡语）：用户 11-28 定了先不做，等主线加完语种。所有纠偏数据目前只有中文；amphiondata 的 `scripts/context_correction` 流程结构上不绑语种，但提示词、分类和质检 ASR 是中文专用的。

<!-- execution-status -->

执行状态：completed；质量状态：未核实。

开始：未知；结束：未结束；退出码：未知。

<!-- /execution-status -->
