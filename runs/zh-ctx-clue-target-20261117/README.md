# 上下文纠偏 think 改成“线索 + 目标”（2026-11-17）

**状态**：2026-11-17 前两组的配置和数据就绪，隐式题训练数据在生成，等 ACP 空位（AmphionASR 合计最多 2 个节点，当时都被其他会话占用）。

**要回答的问题**：同事反馈，纠错时在 CoT 里先定好线索、再指明要改的目标，2B 的文本 LLM 能力也提升很大。
1. 把第二轮的 think 改成这种写法，只换写法、不加新数据，上下文纠偏能不能在训练没见过的说法和真实录音上更好？
2. 补上隐式题训练数据后，think 里写出语义线索和目标，是否比只给数据、think 写固定句更好？

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
