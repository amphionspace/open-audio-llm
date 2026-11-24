# 公开 ASR 榜单调研：AmphionASR 怎么打、还差什么（2026-11-23）

**结论：现在不建议直接提交。** 最值得打的是 Hugging Face Open ASR Leaderboard（关注度最高，中文 11-13 刚上线）。但按 amphion-eval 的数字，AmphionASR-1.7B 在公开测试集上与底座 Qwen3-ASR-1.7B 基本持平、英文略差；AISHELL-1/3 上异常低的错误率来自训练文本与测试文本重合。直接提交排名不突出，还会被榜单的过拟合分析和社区质疑。应先去重、补齐与榜单同口径的评测、确定可公开的发布版权重，再提交。

## 1. 榜单现状

| 榜单 | 现状（核对于 2026-11-23） | 怎么上榜 | 建议 |
| --- | --- | --- | --- |
| [Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard)（Hugging Face） | v1.16.2，2026-11-20 更新。英文主榜、多语种榜（8 种语言，含中文）、长音频榜、私有集榜，另有“过拟合检测”页 | 向 [GitHub 仓库](https://github.com/huggingface/open_asr_leaderboard) 提 PR（评测脚本、Dockerfile、`submit_jobs.sh`），在 HF Jobs 上用 1×H200 跑；维护方再跑私有集。开源模型须在 HF Hub 公开权重，模型卡写明许可证；英文默认集平均 WER 须低于 10% | **主攻** |
| [Artificial Analysis STT](https://artificialanalysis.ai/speech-to-text) | AA-WER v2 = AA-AgentTalk 50% + VoxPopuli-Cleaned 25% + Earnings22-Cleaned 25%，只测英文；开源权重第一为 Voxtral Small（约 2.8%，据检索摘要，未逐页核对） | 需要可公开调用的 API | 暂不投入 |
| [SpeechIO](https://github.com/SpeechColab/Leaderboard)（SpeechColab） | 排名表停在 2025.01，最后一次提交 2025-03-29；排名里只有云 API，没有开源模型 | 提交 docker 模型镜像，由其流水线打分 | 不投入；其公开的 ZH00001–00026 可自测后写进技术报告 |
| 技术报告对比表 | 中文社区的事实标准：AISHELL-1/2、WenetSpeech、KeSpeech、SpeechIO、FLEURS、WenetSpeech-Yue 等。例：2026-11 混元 [Hy ASR 3.0](https://arxiv.org/abs/2611.14908)，报告里写明做了严格的数据去污染 | 随开源权重一起发布 | 配合发布 |

### Open ASR 榜单细节（v1.16.2）

- **英文主榜**默认 9 个集合：AMI-Cleaned、Earnings22-Cleaned-AA-chunked、Gigaspeech-Cleaned、SPGISpeech、URGENT24（加噪）、Voice Arena Monsoon（印度口音对话）、VoxPopuli-AA-Cleaned，以及私有集（朗读、对话）。LibriSpeech 已在 v1.14.0 移出默认集。
- **多语种榜**：中文、德、法、意、西、葡、印地、荷兰语；8 种语言都有结果才参与排名，缺语言的列在表尾，不排名。中文用 FLEURS 和 AISHELL-1，按 CER 计分；规范化器 `normalizer/chinese_normalizer.py` 做 NFKC、繁转简、阿拉伯数字转中文数字（`wetext`）、去标点。
- **过拟合检测**：一项看模型是否复现 VoxPopuli 原始参考文本里的错误，一项看拼写习惯是否跟着测试集走；私有集用来衡量没见过的数据。
- **Qwen 系评测脚本**（`qwen/run_eval.py`、`run_eval_ml.py`）：`qwen_asr` 包的 `Qwen3ASRModel.from_pretrained`，bf16，batch 16，传语言提示，不传 prompt，`max_new_tokens` 英文 512、多语种 256。与 Qwen3-ASR 同结构的权重可以直接复用这套脚本，`oruk/orukqwen-asr-1.7b` 等微调模型就是这样上榜的。

### 当前排名摘录

英文主榜（平均 WER %，越低越好）：

| 排名 | 模型 | 平均 WER | 备注 |
| --- | --- | --- | --- |
| 1 | zoom/scribe_v2.1_pro | 4.87 | API |
| 6 | Musci-research/Musci-ASR-qwen-1019-hf | 5.70 | 开源，名称显示基于 Qwen-ASR |
| 12 | oruk/orukqwen-asr-1.7b | 5.97 | 开源，名称显示为 Qwen3-ASR-1.7B 微调 |
| 15 | nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-FP8 | 6.12 | 开源 |
| 17 | Qwen/Qwen3-ASR-1.7B-hf | 6.15 | **我们的底座** |
| 22 | ibm-granite/granite-speech-5.0-2b | 6.68 | 开源 |

多语种榜（8 种语言平均错误率 %，中文列为 CER）：

| 排名 | 模型 | 平均 | 中文 |
| --- | --- | --- | --- |
| 1 | microsoft/azure-speech-09-2026 | 3.27 | **1.89** |
| 5 | nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B | 4.50 | 3.16 |
| 7 | sprag/symphony（API） | 4.64 | 1.94 |
| 12 | Qwen/Qwen3-ASR-1.7B-hf | 5.94 | 2.12（开源模型里中文最好） |
| 17 | openai/whisper-large-v3 | 7.76 | 4.78 |

## 2. AmphionASR 当前位置

数据：amphion-eval `asr-benchmark@v9-20261113`，打分口径版本 4，部署栈（vLLM）推理。AmphionASR 取 v9 上最新的 1.7B 记录 `ts-robust-062-avg3`（2026-11-16）；11.17 交付包还没有 v9 记录。对照为同口径下的 Qwen3-ASR-1.7B 和 18 个开源基线中的最好值。

| 测试集 | AmphionASR | Qwen3-ASR-1.7B | 开源最好 |
| --- | --- | --- | --- |
| WenetSpeech 网络 | 5.46 | **4.49** | 4.49 |
| WenetSpeech 会议 | 5.63 | **4.69** | 4.69 |
| AISHELL-1 | 0.35 ⚠️ | 1.47 | 0.99（MiMo） |
| AISHELL-2 | 2.62 | 2.55 | 2.15（Ming-tiny） |
| AISHELL-3 | 0.77 ⚠️ | 1.91 | 1.09（Ming-tiny） |
| THCHS-30 | 3.57 | 2.86 | 1.37（MiMo） |
| Common Voice 中文 | 4.97 | 5.05 | 4.80（Ming-tiny） |
| 中文平均（7 个集） | 3.14 | 3.15 | 2.92（Ming-tiny） |
| KeSpeech 平均 | 5.59 | 5.52 | 4.23（Ming-tiny） |
| 英文平均（LibriSpeech×2、MLS、GigaSpeech、CV） | 5.43 | 5.10 | 4.38（Ming-tiny） |
| MLS 英文 | 5.61 | 4.59 | 3.24（Ming-tiny） |
| TAL-CS 中英混说 | 7.87 | 7.47 | 5.52（Ming-tiny） |
| 四川话平均 | 15.75 | 12.85 | 12.85 |

警务业务集（指令、术语、口音合成、录音鲁棒性）上 AmphionASR 大幅领先（例如录音鲁棒性平均 1.57 对 5.31），但这些是同领域训练的业务集，公开榜单没有对应项。

## 3. 待优化点（按优先级）

### 3.1 训练文本与测试文本重合（必须先修）

AISHELL-1 0.35%、AISHELL-3 0.77% 比任何公开报告低 3 倍左右，原因是训练集里有测试句原文。

做法：用 `ts-robust-062-avg3` 训练配置里的全部中文数据源（60 个），把每条训练文本去标点、转小写后，和 v9 各中文测试集的参考文本做整句比对。测试句原文在训练集出现过的算“重合”。

| 测试集 | 重合句占比 | 主要来源 |
| --- | --- | --- |
| AISHELL-1 | **96.5%**（6925 / 7176） | `aishell2` 训练集（6922 句），另有 `ts_scene_pool`、`hw_sft` |
| AISHELL-3 | **61.8%**（15316 / 24773） | `aishell2` 训练集（15191 句）、`hw_sft`、`ts_scene_pool` |
| MagicData | **56.2%**（13633 / 24279） | `hw_sft`（取自 MagicData 训练集音频，13549 句） |
| WenetSpeech 网络 | 2.6% | `wenetspeech_clean`，多为短的常用句 |
| AISHELL-2 | 2.5% | 零散 |
| Common Voice 中文 | 0.3% | 零散 |
| THCHS-30 | 0% | — |

训练用的都是各语料的 train 部分，没有测试集音频（`hw_sft` 的源路径都在 `train/` 下）。问题出在语料本身：这几个朗读语料的 train 和 test 共用朗读稿，AISHELL-2 的朗读稿又覆盖了 AISHELL-1 的测试句。

按是否重合分开算 CER（%，同一批预测，近似计算；全量值与榜单相差不到 0.2）：

| 模型 | AISHELL-1 重合 / 不重合 | AISHELL-3 重合 / 不重合 | MagicData 重合 / 不重合 |
| --- | --- | --- | --- |
| AmphionASR `ts-robust-062-avg3` | **0.21** / 3.74 | **0.20** / 2.79 | 0.77 / 3.46 |
| AmphionASR v3 `34479` | 0.37 / 3.66 | 0.41 / 2.61 | 0.88 / 3.13 |
| Qwen3-ASR-1.7B | 1.37 / 4.89 | 1.43 / 3.65 | 1.47 / 3.46 |
| Ming-3.0-asr-tiny | 0.97 / 2.95 | 0.78 / 2.36 | 0.79 / 2.92 |
| MiMo-V2.5-ASR | 0.88 / 4.18 | 1.08 / 3.65 | 1.35 / 3.78 |

- 重合的句子上，AmphionASR 比其他模型低 4–6 倍；不重合的句子上处于中游：比 Qwen3-ASR 好，比 Ming-tiny 差。所以 AISHELL-1/3 的领先主要是**记住了文本**，不是声学能力。不重合子集只有 251 句（AISHELL-1），本身偏难，所有模型都在 3–5%。
- MagicData 上 Ming-tiny 的重合子集也是 0.79，差距不大，我们在这个集上并不特殊。
- 早期的 v3（34479）已经如此，不是近期训练引入的。
- Open ASR 多语种榜的中文列就包含 AISHELL-1。用现在的模型提交，中文分会被这部分直接拉低，而榜单有专门的过拟合分析页，这种结果很容易被看出来。

修复成本低：AISHELL-2 训练集里文本与 AISHELL-1 或 AISHELL-3 测试句重合的只有 49,247 条、46.4 小时（占 4.9%）。按整句文本把这些条目和 `hw_sft`、`ts_scene_pool` 等派生集里的同类条目排除，形成新的数据版本即可。对外报数字时，要么用去重后训练的模型，要么同时给出不重合子集的结果并说明情况。

### 3.2 英文不如底座，英文主榜没有优势

- v9 英文平均 5.43，底座 5.10；MLS 5.61 对 4.59、GigaSpeech 8.96 对 8.68、Common Voice 7.61 对 7.30。
- 底座在英文主榜排第 17（6.15），已有两个 Qwen-ASR 微调排在前面（5.70、5.97）。按现在的差距，AmphionASR 大概率排在底座之后。
- 英文主榜 9 个默认集里，我们只测过 GigaSpeech；AMI、Earnings22、SPGISpeech、URGENT24、VoxPopuli、印度口音对话和私有集都没测过，无法预估名次。

### 3.3 评测口径与榜单不一致

- FLEURS 中文：我们的 `multilingual-asr-benchmark@v2` 上 Qwen3-ASR-1.7B 为 6.24% CER，榜单上它的中文列（FLEURS 与 AISHELL-1 的平均）是 2.12%。抽样看，我们用的 FLEURS 参考文本带有不读出来的括注，例如“埃尔多安（Recep Tayyip Erdoğan）”“邓迪大学（University of Dundee）”，模型不读就算错。
- 数字规范化也不同：榜单用 `wetext` 把阿拉伯数字转成中文数字；我们用 cn2an，并去掉“呃、啊”和儿化音。
- 英文两边都用 Whisper `EnglishTextNormalizer`，口径一致。

### 3.4 多语种覆盖不够

多语种榜要求 8 种语言都有结果才排名。AmphionASR 没在 `multilingual-asr-benchmark@v2` 上评测过；德、法、意、印地、荷兰语没有训练，后续训练很可能已经让这些语言相对底座退化。提交后最多以“未排名、只有中文列”的形式出现。

### 3.5 发布形态

- 开源上榜需要在 HF Hub 公开权重。评测脚本按 `qwen_asr` 格式加载，需要导出合并 LoRA 后、能被 `qwen_asr` 直接加载的 checkpoint（不带额外的 MTP 等模块），并固定 revision。榜单跑的是 transformers 推理，我们部署和评测用 vLLM，提交前要先做两种实现的一致性核对。
- 训练数据含客户警务数据（`police_asr_unified`、`police_terms_single_train` 等）和仅限学术使用的语料。公开权重前需要确认能否发布，或者训练一个不含这些数据的发布版。这一项需要负责人决定。

## 4. 建议路线

1. **准备（不需要训练）**：对所有公开测试集做训练集去重核查；在 amphion-eval 里登记一个与 Open ASR 同数据、同规范化器的评测集（英文 7 个公开集、中文 FLEURS + AISHELL-1、多语种）；先用 Qwen3-ASR-1.7B 复现榜单数字（6.15 / 2.12），确认口径对齐，再测 11.17 交付包和两个已上榜的 Qwen 微调模型。
   - 2026-11-24 已完成口径对齐和 11.17 评测，见[实验记录](../experiments/2026-11-24-open-asr-leaderboard/README.md)：Qwen3 逐集与榜单差 ±0.11；11.17 英文公开集 6.12（第 25/68）、中文列 2.24。两个 Qwen 微调模型还没测。
2. **决定发布版**：用哪份数据训练可公开的权重（负责人决定）。
3. **训练发布版**：去掉重合文本；守门是英文榜单集不差于 Qwen3-ASR-1.7B，中文保持不退。
4. **提交与发布**：给 `huggingface/open_asr_leaderboard` 提 PR（复用 `qwen/` 脚本或新建目录），同时发模型卡和技术报告；中文表格用去重后的数字，附 SpeechIO 公开集、WenetSpeech-Yue/Chuan、KeSpeech。

## 5. 证据与复现

- 榜单数据：Open ASR Space 的 Gradio 配置（`https://hf-audio-open-asr-leaderboard.hf.space/config`，v1.16.2，2026-11-20）；版本历史见 Space 的 `init.py`（经 hf-mirror 读取）。本机直连 huggingface.co 不通，hf.space 和 hf-mirror 可用。
- 评测数字：`ae sync --task asr` 后 `ae results export --task asr`（2026-11-24 06:09 UTC 同步，2220 条记录）。
- 重合核查：[`overlap_scan.py`](2026-11-23-public-asr-leaderboards/overlap_scan.py)（输入：`ts-robust-062-avg3` 的 v9 预测 `run-20261116T034112Z-07b8d145/predictions.jsonl` 和训练配置 `attempts/062/data-effective.yaml`），结果见 [`overlap-ts-robust-062.json`](2026-11-23-public-asr-leaderboards/overlap-ts-robust-062.json)；分组 CER：[`split_cer.py`](2026-11-23-public-asr-leaderboards/split_cer.py)，结果见 [`split-cer.json`](2026-11-23-public-asr-leaderboards/split-cer.json)。比对只做整句完全一致（去标点、小写），没有统计近似重合，所以重合比例是下限。
- COS 备份 `anfeiweng-audiollm-resource-1373259071:open-audio-llm/research/2026-11-23-public-asr-leaderboards/`（已上传核对）：
  - `open-asr-leaderboard-space-config-v1.16.2.json`（552 KiB，榜单全部表格快照）
  - `open-asr-leaderboard-space-init-v1.16.2.py`（82 KiB，版本历史与数据源配置）
  - `ae-results-export-asr-20261124T0609Z.md`（1.6 MiB，amphion-eval ASR 汇总导出）
  - `overlap-hits-ts-robust-062.json`（3.2 MiB，重合文本及其训练来源）
- 来源阅读情况：Open ASR 的 README、PR 模板、`qwen/` 脚本和中文规范化器读了原文，私有集流程只读了 README 的说明；SpeechIO 读了 README、HOW_TO_SUBMIT 和提交记录；Hy ASR 3.0 读了 arXiv HTML 正文表格；Artificial Analysis 只有检索摘要。
