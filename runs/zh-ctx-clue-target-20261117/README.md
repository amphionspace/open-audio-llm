# 上下文纠偏 think 改成“线索 + 目标”（2026-11-17）

**状态**：2026-11-18 四组都训练、评测完成，ACP 节点已释放。
- **第 1 问**：只换写法，主判定不满足，保留第二轮写法。
- **第 2 问**：
  - 补隐式题数据有效，真实录音上也有效。
  - think 里写语义线索 + 目标只在合成题上更好，真实录音上与固定句没有差别。
- 见“结果：第 1 问”和“结果：第 2 问”。
- **第 3 问**（2026-11-20 加）：用户确认隐式题数据进上下文纠偏的 SFT 配方，加第 5 组 template-implicit-noexplain 确认“第二轮写法 + 隐式题固定句”这个组合。判定已在训练前写定，见“判定”的第 3 问；训练、评测进行中。

**要回答的问题**：同事反馈，纠错时在 CoT 里先定好线索、再指明要改的目标，2B 的文本 LLM 能力也提升很大。
1. 把第二轮的 think 改成这种写法，只换写法、不加新数据，上下文纠偏能不能在训练没见过的说法和真实录音上更好？
2. 补上隐式题训练数据后，think 里写出语义线索和目标，是否比只给数据、think 写固定句更好？
3. 保留第二轮的 think 写法，只补隐式题数据（think 写固定句），能不能拿到第 2 问里补数据的收益？这是准备进上下文纠偏 SFT 配方的组合。

依据是 09 的诊断（`../zh-spelling-think-grpo-20261115/README.md` 的“think 里写了什么”，在 oal-think-exp 分支上）：
- 现在的 think 一行只写一个音、一个字：“`<拼音>` 这个音，说话人说是「`<原话>`」，写作「`<字>`」”。不写这个字落在转写的哪个词里，也不写不该写成什么。
- 线索基本都找到了（独立组 96–99%），失败在后面：
  - 逐字说明每题有 2 个以上目标字，think 只写全 1%；
  - think 写对了字，答案也不一定照着写，拆字题里有 17% 没照着写。
- 隐式题没有推理步骤，Emilia 上还有一半 think 错套了说明模板。训练数据里没有隐式题，要先补数据（第 2 问）。

## 设计

| | 对照组 template | 实验组 clue-target |
|---|---|---|
| 起点 | 11.06 交付版（GRPO v10 插值 50%），同第二轮 | 同左 |
| 训练 | 第二轮配方：1 节点 8 卡全参 3000 步，学习率 3.55e-6，预热 300 步，余弦降到 1e-6，seed 42 | 同左 |
| 联合 MTP | 权重 0.1、3 步、衰减 0.8、lr 1e-4（AGENTS.md“MTP”）。起点没有 `mtp/`，从 11.17 交付包的 `model/mtp/` 初始化 | 同左 |
| 数据 | 第二轮的数据配置原样 | 同左，只换 12 个 think 文件 |
| think | 第二轮的 think 文件 | 同一批行改写成线索 + 目标（`context_correction_clean_zh/versions/think-v3-20261117`） |
| 代码 | `shared/code-v1`（main 0a1b117） | 同左 |

第 2 问再加两组，都在 clue-target 上加同一份隐式题数据 `context_correction_synth_zh@implicit-v1-20261117`（见下文“隐式题数据”），只差隐式题的 think：

| | clue-target-implicit | clue-target-implicit-noexplain |
|---|---|---|
| 隐式题的 think | 线索 + 目标，线索是语义线索：`线索：说话人说「考完这门就放寒假了」。目标：写「期终」，不写「期中」。` | 同一批记录，think 开时只写固定句“这段没有说明字怎么写。” |
| 其余 | clue-target 的数据加隐式题数据（train 和 train_ts）。加法同第二轮加合成数据：每条在 3000 步里约见 3 次，think 开的概率 0.5 | 同左 |

- 这两组比的是：数据相同时，think 里写不写隐式题的线索和目标。
- 各自与 clue-target 比，看的是补隐式题数据的效果。
- 配置要等数据构建完、知道条数后才能写（`scripts/make_configs.py --implicit`），任务也在那时加进 `experiment.yaml`。实验入口遇到配置还不存在的任务会直接报错。

- **为什么要新的对照组**：
  - 2026-11-15 起 SFT 必须开联合 MTP，第二轮没有开，所以不直接和第二轮比。
  - 两组都开 MTP，只差 think 写法。
  - code-v1 和第二轮的 code-v2 在数据和训练路径上只差联合 MTP 学习率的修复（#164）和 vLLM 服务代码。`src/open_audio_llm/data/` 逐文件相同。
- **两组看到的样本相同**：
  - 每条样本开不开 think，由样本自己的种子决定（`data/think.py` 的 `ThinkTable.choose`），与 think 文本无关。
  - 两组的 think 表核对过：40,434 行，记录 ID 和 think 开的集合（8,049 条）逐条相同；实验组的 8,049 条全部是新写法。
- **不是 GRPO 后的模型**：两组都只做 SFT，比的是 think 写法对上下文纠偏的影响，不判断能否交付。写法有用时，再按“SFT 后做一轮 GRPO”的流程走。

### think 写法

`线索：说话人说「<原话>」。目标：写「<目标>」，不写「<另一种写法>」。`

| 来源 | 第二轮 | 本实验 |
|---|---|---|
| 合成（5,862 条） | `yuan 这个音，说话人说是「锦源路的源是水源的源，不是花园的园」，写作「源」。` | `线索：说话人说「锦源路的源是水源的源，不是花园的园」。目标：写「锦源」，不写「锦园」。` |
| 真实，清洗时改过字（229 行） | `ke 这个音，说话人说是「是纽约客的那个客」，写作「客」。` | `线索：说话人说「是纽约客的那个客」。目标：写「客」，不写「课」。` |
| 真实，原转写已对（2,367 行） | `fu 这个音，说话人说是「就是幸福的福」，写作「福」。` | `线索：说话人说「就是幸福的福」。目标：写「福」。` |

- **合成数据**：
  - 目标取记录的目标词（`labels.context_correction.segments` 里 role 为 target 的整个词，如店名、人名），“不写”取它的同音写法。
  - 每条只有一个目标词；同名一致性题的名字出现两次，只写一行。
- **真实数据**：
  - 清洗决定里没有标出这个字落在转写的哪个词里，所以目标仍是原来那一行的字。
  - 只有清洗时改过字的窗口知道原转写写成了什么，这部分写“不写”，其余省略。
- **改动范围**：
  - 原话沿用原行，不重新切。
  - 去掉了拼音：同事的写法没有拼音，“不写”已经给出了同音的另一个字。
  - 没有说明的固定句“这段没有说明字怎么写。”不变。
- **脚本**：`scripts/make_think_v3.py`。逐条对照见数据目录的 `samples.jsonl`：清洗时改过字的全部列出，另外每个来源随机抽 20 条。
- **数据目录**：`/workspace/data/datasets/context_correction_clean_zh/versions/think-v3-20261117/`。
  - `state.json` 是 prepared；按来源的行数在 `summary.json`。
  - 这是实验用的变体，没有登记 catalog，写法被采用时再登记。

### 隐式题数据

`context_correction_synth_zh@implicit-v1-20261117`，运行目录 `/workspace/workspace/mingdong/data/amphiondata/ctx-implicit-zh-20261117/`（试跑、全量、拒绝原因抽查见那里的 README）。占位登记已合并（audio-data-contract #261、#262），已同步到 registry。
- **流程**：同 train-v1（amphiondata `feat/context-correction-implicit-train-data`，叠在 MR !28 上），只做隐式语义。
  - 规则同评测集生成器：被测词是常见词语、放在前面；两条文字在被测词之前完全相同；不要求 `prior_matches`；复核要判定前半句两种写法都说得通。
  - 出题和复核改用火山方舟（百炼仍欠费）：doubao-seed-2-1-pro 出题，deepseek-v4-pro 复核。用户 2026-11-17 确认模型同源不是问题，只要话题种子设好就不会重复。
- **话题种子**：15 个话题、75 个子话题，每个规格先抽话题再抽子话题。与评测集 v1、v2 的场景，以及 train-v1 的场景都不重合。
- **去重**：评测集 v1、v2 的 4,083 个被测词进回避列表；写法对或朗读文本与任一条相同的进隔离区。同一写法对最多收 3 对。
- **不用盲解筛选**：试跑中盲解拒掉的题一半以上是解题模型自己写错，而且集中在少见写法上，所以只靠复核把关。
- **语音和 think**：音色、TTS、质检同 train-v1。think 用 amphiondata `think.py --style clue-target`，在 train-v1 上的输出与本实验的 think-v3 合成部分 5,901 行逐条相同。

第 3 问加一组，从第 2 问的结果来：真实录音上语义线索没有额外收益（Emilia 线索只写对 23%），所以试最简单的组合。

| | template-implicit-noexplain |
|---|---|
| think | 第二轮的 think 文件（同 template） |
| 隐式题 | 同第 2 问两组：`context_correction_synth_zh@implicit-v1-20261117`，train 1,685 条、train_ts 1,684 条，权重同 clue-target-implicit-noexplain，约占样本 1.79% |
| 隐式题的 think | 固定句“这段没有说明字怎么写。”（`think-no-explanation.jsonl.gz`，同 clue-target-implicit-noexplain） |
| 其余 | 同 template：起点、配方、联合 MTP、seed、代码 code-v1；端口 29866 |

- 与 template 比：只差隐式题数据和它的 think 文件（数据配置结构化比对核过）。
- 与 clue-target-implicit-noexplain 比：只差显式说明题的 think 写法（think-v2 / think-v3），隐式题数据和权重相同。
- 配置：`scripts/make_configs.py --implicit template-implicit-noexplain`。

## 评测

各组的 checkpoint-3000 都测上下文纠偏评测集 v2（`asr/context-correction-zh@v2-20261117`），think 关、开各一遍。
- 口径和 09 测 avg3 时相同：ae 0.14.0-eedcf15，新 TS 提示，batch 32，贪心解码。
- 服务：vLLM 0.32（`vllm-serving-032`，main 的 TS 服务需要它）。用 checkpoint 自带的 `mtp/` 做投机解码，EAGLE k=2。校验是贪心的，不改变输出。
- 对比：
  - 配对 bootstrap 用 `ae open-audio-llm compare`；
  - think 写了什么用 `scripts/ctx_v2_think_clue_target.py`，从 09 复制，两种写法都能读，在 09 的 avg3 结果上复跑主表逐行相同。

## 判定（训练前写定）

都看 think 开，比较实验组与对照组，配对 bootstrap 95% CI：
- **主判定**：独立合成组的纠正成功率和成对正确率提高（CI 不含 0），并且 Emilia 准确率不下降（CI 下限不低于 −1 个点）。
- **机制核对**：
  - 逐字说明的纠正成功率，以及 think 写全目标字的比例；
  - think 写全目标字时答案也写对的比例；
  - Emilia 隐式题里套用说明格式的比例。
- **不能变差**：保持率和误改率不变差；think 关的纠正成功率不下降。
- **结论**：
  - 主判定满足：写法有效；
  - 不满足：记录结果，保留第二轮的写法。
- **第 2 问**，都看 think 开：
  - clue-target-implicit 与 clue-target-implicit-noexplain 比 Emilia 需纠正的隐式题，以及独立合成组的隐式题，看准确率和纠正成功率；
  - 两组都与 clue-target 比，看补数据的效果；
  - 不能变差：Emilia 原转写已对的准确率、保持率和误改率。隐式题教模型按语义改字，最担心的是乱改。
- **第 3 问**（2026-11-20，训练前写定），都看 think 开，template-implicit-noexplain 与 template 比：
  - **主判定**：独立合成组的隐式题准确率，以及 Emilia 的隐式题准确率（`ae open-audio-llm compare` 的 `implicit_accuracy`，与第 2 问结果表同一指标），都提高（CI 不含 0）。
  - **不能变差**：Emilia 准确率、全部保持率，CI 下限不低于 −1 个点；全部误改率 CI 上限不高于 +1 个点；think 关的全部纠正成功率 CI 下限不低于 −1 个点。Emilia 原转写已对的 703 条没有配对 CI，看 `ctx_v2_mined_groups.py` 的准确率，不低于 template 1 个点以上。
  - 2026-11-20 03:34 UTC（提交 a0e5fa0）改了上面两句的措辞，让每一项都对应比较工具实际输出的指标；改时还没有本组的评测结果。
  - **场景 TS 不能变差**（2026-11-20 03:42 UTC 补写，提交 6095767，本组还在训练）：鼎桥场景 TS（`tdtech/scene-ts@v1-20261028`，think 关）上，本组与 template 比，CER 和目标不在场误出字率都不显著变差（CI 下限不高于 0）。
    - 原因：场景 TS 退化是这条线进交付的卡点（08：第二轮 checkpoint-3000 CER 9.93，交付版 3.35），最可疑的是第二轮的合成 `train_ts`（单人干净录音、目标每条都在场）。隐式题又加了 1,684 条同类的 `train_ts`，所以要确认它不加重退化。
    - 做法：`scripts/acp-60-scene-ts.sh` 在同一节点的 GPU 2/3 上测 template 和本组的 checkpoint-3000，与本组的上下文纠偏评测同时跑；配置由 `scripts/make_scene_ts_configs.py` 生成，服务同本实验的评测（vLLM 0.32、自带 mtp/），测试配置同 08。
    - 即使通过，配方仍带着第二轮的场景 TS 退化，进交付前要另外解决。
  - **参考**：与 clue-target-implicit-noexplain 比 Emilia 准确率和 Emilia 隐式题准确率，看真实录音上第二轮写法是否够用；机制核对看 Emilia 需纠正隐式题里套用说明格式的比例（template 57%）。
  - **结论**：
    - 都满足：上下文纠偏 SFT 配方采用“第二轮 think + 隐式题（固定句）”，即本组的数据配置；隐式题的固定句 think 文件随配方登记。
    - 不满足：记录结果，不定配方，汇报后再决定。

## 结果：第 1 问（template 与 clue-target，2026-11-18）

**结论：主判定不满足，按训练前写定的规则保留第二轮的写法。**
- 线索 + 目标让答案更照着 think 写，独立组纠正成功率 +2.5 个点；但成对正确率的提升不显著，Emilia 准确率的 CI 下限低于 −1。
- 瓶颈仍在 think 写不出对的字，这不是写法能解决的。
- 没有说明的隐式题被新写法套用得更多，Emilia 隐式题因此变差。第 2 问的两组正是针对这一点。

四个评测 run 都完整：9,892/9,892，没有失败样本。
- 差 = clue-target − template。
- 配对 bootstrap 用 AmphionEval !158 的 `ae open-audio-llm compare`，按成对题整对抽样，共 3,821 簇，2,000 次。
- `*` 表示 95% CI 不含 0。

| think 开 | template | clue-target | 差 [95% CI] |
|---|---|---|---|
| 独立合成组 · 纠正成功率 | 49.0 | 51.5 | +2.5 [+0.5, +4.8]* |
| 独立合成组 · 成对正确率 | 42.0 | 43.8 | +1.8 [−0.3, +4.1] |
| 与训练文本不重合（unseen）· 纠正成功率 | 47.5 | 50.3 | +2.8 [+0.4, +5.2]* |
| Emilia · 准确率 | 71.5 | 70.6 | −0.8 [−1.9, +0.3] |
| 全部 · 保持率 | 86.0 | 86.1 | +0.1 [−1.2, +1.4] |
| 全部 · 误改率 | 3.3 | 3.6 | +0.4 [−0.4, +1.1] |
| 全部 · 隐式题准确率 | 59.0 | 57.6 | −1.4 [−2.4, −0.4]* |
| 同源合成（v1）· 纠正成功率 | 77.8 | 75.9 | −1.9 [−4.8, +1.1] |
| 播客（127 条，来源在训练里）· 准确率 | 83.5 | 78.0 | −5.5 [−10.2, −1.6]* |

- **think 关不变**：
  - 纠正成功率 47.2 → 46.9（−0.3 [−1.3, +0.7]）；
  - 独立组 37.2 → 36.9（−0.4 [−1.6, +0.9]）；
  - 保持率 84.0 → 84.0。

**机制核对**（think 开；`scripts/ctx_v2_think_clue_target.py`，表在 `tasks/02-eval/compare/think-clue-target-{template,clue-target}.md`）
- **答案更照着 think 写**：
  - think 写全目标字时，答案也写对的比例在独立组各子类上从 80–95% 升到 94–100%。例如拆字 88 → 100、专名指代 83 → 99、同名一致性 84 → 100、组词 81 → 96。
  - 09 发现的“think 写对了、答案不照着写”基本消除。
- **think 写出对的字的比例没变**：
  - 拆字 25 → 30%，偏旁结构 44 → 42%。
  - 线索对上、字全写错的仍多：拆字 70%，偏旁 54%，对话问答 45%。
  - 这是部件拼字和字形知识的问题，不是写法问题。
- **逐字说明**：think 写全目标字从 3% 升到 23%，但纠正成功率 25 → 24，没有变化。
- **隐式题被套用说明格式**：
  - Emilia 需纠正的隐式题里，think 写说明格式的从 57%（361/629）升到 81%（509/629）。这部分 think 开的准确率 64.3 → 62.9。
  - 新写法要求先写“目标：写某字”，没有说明时也会硬写一个目标。

**Emilia 按原转写是否需要纠正拆开**（think 开；`scripts/ctx_v2_mined_groups.py`，表在 `tasks/02-eval/compare/mined-groups-ab.md`）

| | 条数 | template | clue-target |
|---|---|---|---|
| 需纠正 · 显式说明 | 516 | 52.1 | 52.3 |
| 需纠正 · 隐式 | 629 | 65.5 | 64.1 |
| 原转写已对 | 703 | 91.3 | 90.0 |

**证据**
- **评测 run**：`whai:open-audio-llm/runs/zh-ctx-clue-target-20261117/ae-runs/{template,template-think,clue-target,clue-target-think}/runs/`
  - template：`run-20261118T110003Z-ab2fd20e`（think 关）、`run-20261118T110622Z-f37d2312`（think 开）
  - clue-target：`run-20261118T110104Z-c847774a`（think 关）、`run-20261118T110726Z-bee67643`（think 开）
- **配对 bootstrap**：`tasks/02-eval/compare/cmp-template-cluetarget-{off,on}.json`。
  - 已装的 ae 0.14.0 和 0.15.0 都还没有 !158 的上下文纠偏指标。
  - 所以用合入 main 的同一份代码（`amphion-eval-ctx-compare`，beb1f99），配 0.14.0 的环境运行，与 09 相同。
- **训练**：
  - [template](https://wandb.ai/1016097967-amphion/open-audio-llm/runs/AmphionASR-ctx-template-think-r2recipe-mtp-from-v10interp050-3k-1node-codev1-001)、[clue-target](https://wandb.ai/1016097967-amphion/open-audio-llm/runs/AmphionASR-ctx-clue-target-think-r2recipe-mtp-from-v10interp050-3k-1node-codev1-001)。
  - checkpoint 在 `whai:open-audio-llm/runs/zh-ctx-clue-target-20261117/tasks/01-train-{template,clue-target}/attempts/001/artifacts/training/`。

## 结果：第 2 问（隐式题两组，2026-11-18）

**结论**
- **补隐式题数据有效**：两组都比 clue-target 好，真实录音上也好。
  - 独立合成组的隐式题准确率：think 写线索的组 +15.2 个点，写固定句的组 +9.8 个点。
  - Emilia 隐式题准确率：+2.7 和 +2.2 个点。
  - 原转写已对的 Emilia 条目没有被乱改。
- **think 写语义线索 + 目标，只在合成题上更好**：
  - 与固定句组比，独立合成组隐式题准确率 +5.5、成对正确率 +10.9。
  - Emilia 上没有差别（+0.4，CI 含 0）。
  - 合成题的语义线索 96% 能写对，Emilia 只有 23%：真实录音的语义线索与合成题差别大。

四个评测 run 都完整（9,892/9,892，无失败），配对 bootstrap 同第 1 问。

**think 开**

| | clue-target（无隐式题数据） | 隐式题 · 固定句 | 隐式题 · 线索 + 目标 | 线索 − 固定句 | 线索 − clue-target |
|---|---|---|---|---|---|
| 独立合成组 · 隐式题准确率 | 59.4 | 69.2 | 74.6 | +5.5 [+2.3, +8.6]* | +15.2 [+11.8, +18.7]* |
| 独立合成组 · 隐式题成对正确率 | 24.1 | 40.1 | 51.0 | +10.9 [+5.0, +17.1]* | +26.9 [+20.5, +33.2]* |
| 独立合成组 · 纠正成功率（显式说明） | 51.5 | 51.9 | 52.2 | +0.4 [−1.2, +1.9] | +0.7 [−1.2, +2.6] |
| 独立合成组 · 保持率 | 84.2 | 83.4 | 82.4 | −0.9 [−2.1, +0.5] | −1.8 [−3.2, −0.4]* |
| 全部 · 误改率 | 3.6 | 3.8 | 3.7 | −0.1 [−0.6, +0.4] | +0.1 [−0.5, +0.6] |
| Emilia · 隐式题准确率 | 65.6 | 67.9 | 68.3 | +0.4 [−1.1, +2.0] | +2.7 [+0.6, +4.9]* |
| Emilia · 准确率 | 70.6 | 71.7 | 71.9 | +0.2 [−0.7, +1.1] | +1.3 [+0.2, +2.5]* |
| Emilia · CER | 5.0 | 4.4 | 4.5 | +0.1 [−0.2, +0.4] | −0.6 [−1.0, −0.2]* |

- **固定句组 − clue-target**：
  - 独立合成组隐式题 +9.8 [+7.2, +12.8]*；
  - Emilia 隐式题 +2.2 [+0.3, +4.4]*；
  - Emilia 准确率 +1.1 [−0.1, +2.2]。
- **think 关也有效**，补数据的收益不只在 think 开时出现。隐式题 · 线索组 − clue-target：
  - 全部纠正成功率 +2.3 [+1.2, +3.4]*，成对正确率 +1.8 [+0.4, +3.3]*；
  - 独立合成组隐式题准确率 +8.4*。
- **两组隐式题数据的 think 关几乎相同**：线索 − 固定句，全部纠正成功率 +0.1，隐式题准确率 −0.1。数据相同时，think 的写法只影响 think 开。

**Emilia 按原转写是否需要纠正拆开**（think 开；`tasks/02-eval/compare/mined-groups-implicit.md`）

| | 条数 | template | clue-target | 隐式题 · 固定句 | 隐式题 · 线索 + 目标 |
|---|---|---|---|---|---|
| 需纠正 · 显式说明 | 516 | 52.1 | 52.3 | 52.3 | 52.3 |
| 需纠正 · 隐式 | 629 | 65.5 | 64.1 | 66.5 | 67.1 |
| 原转写已对（不能乱改） | 703 | 91.3 | 90.0 | 91.3 | 90.9 |

**think 写了什么**（`tasks/02-eval/compare/think-clue-target-<组>.md`）

| | template | clue-target | 隐式题 · 固定句 | 隐式题 · 线索 + 目标 |
|---|---|---|---|---|
| 独立合成组隐式题：纠正成功率 | 30 | 32 | 48 | 62 |
| 独立合成组隐式题：think 线索对上 | 5 | 38 | 0 | 96 |
| Emilia 需纠正隐式题：think 线索对上 | 21 | 36 | 15 | 23 |
| Emilia 需纠正隐式题：套用说明格式的比例 | 57 | 81 | 40 | 51 |

- **补数据压住了乱套说明格式**：第 1 问里的副作用（81%）降到 40–51%。
- **线索组在合成题上几乎都能写出语义线索**（96%）。写全目标字时，答案也写对的比例是 98%。
- **Emilia 上写不出线索**：线索对上的只有 23%，所以和固定句组没有差别。

**与对照组 template 比（训练前没有写定，只作参考）**：隐式题 · 线索 + 目标 − template，think 开。
- 独立合成组纠正成功率 +3.2 [+0.9, +5.7]*，隐式题准确率 +14.6*。
- Emilia 准确率 +0.4 [−0.7, +1.5]。
- 误改率：独立合成组 +0.9 [+0.0, +1.7]*。
- 播客（127 条）准确率 −5.5 [−11.0, −0.8]*。

**证据**
- **评测 run**：`whai:open-audio-llm/runs/zh-ctx-clue-target-20261117/ae-runs/{clue-target-implicit,clue-target-implicit-think,clue-target-implicit-noexplain,clue-target-implicit-noexplain-think}/runs/`。
  - 隐式题 · 线索 + 目标：`run-20261118T144449Z-e5bfcdfa`（think 关）、`run-20261118T145111Z-4973a3d1`（think 开）。
  - 隐式题 · 固定句：`run-20261118T144535Z-22b5125f`（think 关）、`run-20261118T145159Z-564940dc`（think 开）。
- **配对 bootstrap**：`tasks/02-eval/compare/cmp-{noexplain-implicit,cluetarget-implicit,cluetarget-noexplain}-{on,off}.json`，以及 `cmp-template-implicit-on.json`。
- **训练**：
  - [隐式题 · 线索 + 目标](https://wandb.ai/1016097967-amphion/open-audio-llm/runs/AmphionASR-ctx-clue-target-implicit-think-r2recipe-mtp-from-v10interp050-3k-1node-codev1-002)（执行记录 002）；
  - [隐式题 · 固定句](https://wandb.ai/1016097967-amphion/open-audio-llm/runs/AmphionASR-ctx-clue-target-implicit-noexplain-think-r2recipe-mtp-from-v10interp050-3k-1node-codev1-001)。
  - checkpoint 在 `whai:open-audio-llm/runs/zh-ctx-clue-target-20261117/tasks/01-train-clue-target-implicit/attempts/002/` 和 `.../01-train-clue-target-implicit-noexplain/attempts/001/` 下的 `artifacts/training/`。

## 不在本实验里

- **拆字、偏旁结构推错字**：这是部件拼字的知识问题，不是写法问题。
- **多字目标**：逐字说明这类题训练里没有，本实验只看新写法能不能推广过去。

## 启动

```bash
python runs/zh-ctx-clue-target-20261117/scripts/make_think_v3.py   # think 文件（已生成）
python runs/zh-ctx-clue-target-20261117/scripts/make_configs.py     # 两组的训练和数据配置
open-audio-llm experiment run --config runs/zh-ctx-clue-target-20261117/experiment.yaml --dry-run
```

在 ACP 上用 inbox 执行器运行，目录 `/workspace/workspace/mingdong/scratch/acp-clue-target`，两个作业都执行 `scripts/node_chain.sh` 的若干阶段。

`10-chain.sh` 按顺序做三件事：
1. **冒烟测试**：用 11.17 交付包起评测服务（GPU 0/1），每个来源测 2 条，提前发现评测栈的问题。失败不影响训练。
2. **训练**：template、clue-target 先后训练，各占 8 卡，端口 29826 / 29836。开训前先确认节点能经转发连上 W&B。
3. **评测**：两组的 checkpoint-3000 经 ae 模型缓存取回，同时评测，template 用 GPU 0/1，clue-target 用 GPU 2/3。

`20-implicit.sh` 是后半程：
1. 等 10-chain 进入评测阶段（此时 GPU 4–7 空闲），并等开发机上的隐式题出题和选题结束。
2. 在 GPU 4–7 上做隐式题的 TTS（4 个分片）和质检，再构建、写 think 和固定句变体。
3. 等 10-chain 结束，写两组隐式题配置，先后训练，再同时评测。

`submit_when_free.py` 在开发机后台等空位：AmphionASR 运行中的任务（不算 `-free` 空占）少于 2 个，且集群有 8 张空卡时，提交 1 节点任务。节点起来后，它把节点 IP 加进 W&B 转发的放行名单。

## 执行记录（2026-11-17）

- 配置和数据就绪；dry-run 通过。
- 等待空位：当时 AmphionASR 运行中的是 `pt-4tx2jx45`（多语种基准评测）和 `pt-dksj7mub`（GRPO v26），都是其他会话的任务。
- 2026-11-18 06:26 UTC 提交 `pt-0uwyn5vh`（节点 10.119.18.77），节点 IP 已加进 W&B 转发放行名单。
- **冒烟测试通过**：12/12 条成功，ASR 和 TS 服务都是 vLLM 0.32。
- **第一次训练没启动**：
  - `train` 在实验目录下运行，配置却按仓库根目录的相对路径传入，06:35 报“文件不存在”退出。开训前的 W&B 连通检查已经通过。
  - 修正：`train()` 先 `cd` 到仓库根目录再启动，与 dry-run 的用法一致。
- **改用 `30-chain.sh` 续跑**：
  - 10-chain 退出后，`20-implicit` 按设计在 GPU 4–7 上开始构建隐式题数据（06:36）。如果放着不管，它构建完就会先训隐式题两组。
  - 于是停掉它的父进程，让构建单独跑完，由 `30-chain.sh` 接着依次做：等构建结束 → 训练并评测前两组 → 写隐式题两组配置 → 训练并评测这两组。
  - 构建期间 GPU 0–3 空闲约 1.5 小时。

<!-- execution-status -->

执行状态：completed；质量状态：未核实。

开始：未知；结束：未结束；退出码：未知。

<!-- /execution-status -->
- **前两组训练完成**：
  - template 07:58–09:24 UTC，clue-target 09:25–10:54，都是 3000 步、退出码 0，W&B 状态 finished。
  - checkpoint 500–3000 都已上传 whai，带 `mtp/`。
  - MTP 学习率：
    - 日志不按参数组记学习率。
    - 生效参数是 `joint_mtp_lr=0.0001`，代码给 MTP 单独建参数组。
    - 第 70 步时骨干学习率还 ≤ 8.3e-7，MTP 第 1 步准确率已从 0.72 升到 0.80，结束时 0.89，与 MTP 用自己的学习率一致。
- **评测**：10:54–11:12，四个 run 都完整。
- **隐式题两组没启动**：
  - clue-target-implicit 的数据预检报“`context_correction_synth_zh@implicit-v1-20261117:train` 没有 Lhotse 产物映射”：catalog 里这一版还只是占位。
  - 正式登记见 audio-data-contract #271，合入并同步 registry 后重跑。
  - 重跑是新的执行记录（attempt 002），`evaluate()` 改为取最新一次完成的训练记录，不再写死 `attempts/001`。
- **隐式题两组**：
  - #271 合入、同步 registry 后，`40-implicit.sh` 11:42 起训练：隐式题 · 线索 + 目标 11:42–13:10（执行记录 002），隐式题 · 固定句 13:10–14:39。
  - 都是 3000 步、退出码 0，W&B finished，checkpoint 500–3000 带 `mtp/` 已上传 whai。
  - 两组的数据预检都读到隐式题 train 1,685 条、train_ts 1,684 条。
  - 评测 14:39–14:57，四个 run 完整。
- **节点**：14:58 停止 `pt-0uwyn5vh`，释放节点。

## 执行记录：第 3 问（2026-11-20）

- 用户确认：隐式题数据进上下文纠偏的 SFT 配方；加第 5 组确认“第二轮写法 + 隐式题固定句”。
- 配置由 `scripts/make_configs.py --implicit template-implicit-noexplain` 生成。结构化比对：与 template 只差两份隐式题来源和一个 think 文件；与 clue-target-implicit-noexplain 只差 think-v2 / think-v3。dry-run 通过。
- 判定在训练前写进“判定”的第 3 问（提交 1e56062）。
- **ACP**：`submit_when_free.py`（运行目录 `/workspace/workspace/mingdong/scratch/acp-ctx-template-implicit`）03:06 UTC 提交 `pt-keiwg4yv`，提交时本项目没有运行中的节点。03:07 节点 10.119.17.215 起来，IP 已加进 W&B 转发放行名单。
- **训练**：`50-template-implicit.sh` 03:07:48 开训，端口 29866；数据预检通过；W&B 远端 running，`train/loss` 已上传。
- 训练完成后在 GPU 0/1 评测 checkpoint-3000，think 关、开各一遍。链路正常结束后由 `stop_when_done.sh` 停止作业，失败时保留节点排查。
