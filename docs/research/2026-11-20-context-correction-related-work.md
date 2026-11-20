# 上下文纠偏相关工作调研（2026-11-20）

**要回答的问题**：上下文纠偏线（实验 `runs/zh-ctx-clue-target-20261117`，分支 `feat/ctx-clue-target-think`）做完第 3 问后，真实录音上有三个问题。别人怎么造数据、怎么训练、怎么评测，下一步该怎么做？
1. Emilia 显式说明题 think 关只有 44%。多数情况是模型听到了说明、也转写对了，却没有改前面的目标字。
2. 合成隐式题数据只在合成题上有提升（+8 个点），真实录音上没有。
3. 补了隐式题数据后，误改变多（think 关保持率 −1.4，显著）。

**怎么查的**：2026-11-20 分四路检索：
- 上下文偏置 / 热词 ASR；
- 说字与中文同音纠错；
- ASR 的推理、RL 与合成到真实的迁移；
- 用户指定的 arXiv 2606.13464 及同类工作。

2606.13464 和 2609.19765 读了正文，其余主要读 arXiv 摘要页和 HTML 正文。标“未核实”的只读到摘要或二手摘要。之前的会话级上下文 ASR 调研见 [2026-11-04](2026-11-04-context-aware-asr-survey.md)。

## 结论

1. **主流做法是 SFT 之后做 RL**：奖励给关键词加权，同时惩罚漏改和多改（Seed-ASR、CAST、2512.21828、RLBR）。本项目交付版那一轮 GRPO 有五项计划（热词、无语音、漏识、TS 缺席、TS 在场），没有上下文纠偏，而且跑完后纠偏小幅变差（think 开纠正成功率 78.51 → 77.07，p=0.009）。**下一步：在 GRPO 里加上下文纠偏计划，用 think 关，与交付一致。**
2. **问题 1 没有现成方案**：所有工作都把证据放在前一轮对话或热词表里，没有“同一句话里说明在目标之后，听到了却不回头改”的情况。只能靠本项目自己的数据和奖励来解决；GRPO 直接奖励 think 关的最终输出，正好对上。
3. **生僻词靠检索或热词提供，不靠模型自己记**（Apple 2409.15353、CAST、DeRAGEC）。所以不再用合成隐式题去追真实录音上的隐式题。
4. **不加约束的改写会比原始识别更差**（2606.13464、LOGIC）。评测时纠正率要和误改率一起报，RL 奖励要对多改扣分。
5. **大模型的字形、拆字知识普遍薄弱**（2410.09013、Phun-Bench），补救办法都是预训练级的。公开资料里没有中文说字的语音数据集，本项目的真实录音评测集是稀缺资产，只做评测，不进训练。

## 本项目同日的本地定位

详见实验 README 的“真实录音失败归因（2026-11-20）”。要点：
- Emilia 显式说明题，think 关时：
  - 94% 的题模型听出了说明（说明部分拼音转对），但只有 44% 改对，39% 仍写成常用字；
  - 在说明本身就含正确字的题里，仍写常用字的有 61% 是“说明逐字转对了、目标没改”，只有 12% 是改写说明去迁就错字；
  - 拆字、偏旁这类需要字形知识的题占 85/516。
- 真实录音的说明数据已经在训练里（2,187 条），混响和噪声增强也已开着，所以这两样都不是新手段。

## 来源

### RL 与上下文偏置

| 工作 | 日期 | 做法 | 与本项目相关的结果 | 读到 |
| --- | --- | --- | --- | --- |
| [Seed-ASR](https://arxiv.org/html/2407.04675v2)（字节） | 2024-07 | 上下文由 LLM 按转写生成，做 context SFT；RL 奖励用关键词加权的 WER | 去掉 context SFT，严格召回从 80.63% 降到 61.26%；加权 WER 比普通 WER 更能保住召回；没报误触发率 | HTML 正文 |
| [CAST: Think Before You Transcribe](https://arxiv.org/abs/2610.26986)（Qwen3-ASR 系） | 2026-10 | 用 Qwen3-ASR-1.7B 无上下文转写真实录音，挖出实体漏识的样本，再由 LLM 复核；SFT 加 GRPO，奖励为 0.8×词准确率 + 0.2×实体召回；先写一句音频描述再转写 | 评测分四种上下文（只给正确词 / 不给 / 检索 / 全是干扰词）；全干扰时 WER 6.2 对 6.3；Qwen3-Omni 的实体漏识率：不给上下文 44.2，给正确词后 10.3 | 摘要和正文片段 |
| [Contextual Biasing with Hotword Retrieval and RL](https://arxiv.org/html/2512.21828v1)（阿里通义） | 2025-12 | 约 200 万条、以 TTS 合成为主，带与不带热词按 8:1 混合，正负热词各半；GRPO 奖励对漏掉和多出的热词都扣分；检索取 top-2 | 测试集是真实录音（Media、Medical 各 240 条）；Media 关键实体错误率 22.63 → 8.29，通用句子准确率 77.49 → 78.16；top-k 变大反而引入干扰 | HTML 正文 |
| [RLBR](https://arxiv.org/abs/2601.13409) | 2026-01（ICASSP 2026） | SFT 后做 GRPO，奖励偏重偏置词，探索时用到参考转写 | 只在 LibriSpeech 上测；偏置词 WER 下降，整体 WER 没有变差 | 摘要 |
| [Explore the RL for LLM-based ASR and TTS](https://arxiv.org/abs/2509.18569)（通义） | 2025-09 | GRPO，规则奖励为 1−WER，另加幻觉检测 | 数据少、步数少也有提升；数字未核实 | 摘要 |
| [Grounded in Sound](https://arxiv.org/abs/2609.14455) | 2026-09 | 在 WER 奖励之外，加冻结 CTC 模型给的声学奖励 | 只用 WER 奖励时，模型会靠语言先验“猜”，出现没有音频依据的插入；加声学奖励后，AMI 插入错误降 22–28% | 摘要 |
| [BR-ASR](https://arxiv.org/abs/2505.19179)（上海交大） | 2025-05 | 按音素编辑距离 ≤2 定义同音词，训练中同音干扰词比例从 0.01 逐步升到 0.5 | 纯提示偏置在偏置词超过 100 个时严重幻觉 | 摘要 |
| [Qwen3-ASR](https://arxiv.org/abs/2601.21337) | 2026-01 | 上下文放 system prompt；上下文数据只在 SFT 阶段用 | 没讲数据构造，也没有定量评测 | 摘要 |
| [Qwen-Audio-3.0-ASR](https://arxiv.org/abs/2609.07549) | 2026-09 | 二手摘要称：SFT 后接 GRPO，奖励含 CER、热词和幻觉检测 | 摘要页只列了功能，训练细节**未核实** | 摘要 |

### 控制误改

| 工作 | 日期 | 做法 | 结果 | 读到 |
| --- | --- | --- | --- | --- |
| [LOGIC](https://arxiv.org/abs/2601.15397) | 2026-01 | 免训练，在输出 logit 上注入上下文，偏置强度可调 | 11 个 locale 上实体错误平均相对降 9%，误触发只升 0.3%；指出生成式纠错会产生幻觉式过度纠正 | 摘要 |
| [RLLM-CF](https://arxiv.org/abs/2505.24347) | 2025-05 | 先检测错误，再推理纠错，最后验证，避免改动本来正确的字 | AISHELL-1/2、LibriSpeech 上相对降 9–21%，数据集和数字的对应关系未核实 | 摘要 |

### 生僻词与检索纠错

| 工作 | 日期 | 做法 | 结果 | 读到 |
| --- | --- | --- | --- | --- |
| [Apple 拼音检索](https://arxiv.org/abs/2409.15353) | 2024-09 | LLM 先检出实体，按音近检索数据库，再把候选放回提示 | WER 最多降 30.2%，实体错误最多降 73.6% | 摘要 |
| [DeRAGEC](https://arxiv.org/html/2506.07510v1)（POSTECH） | 2025-06（ACL Findings） | 文本侧免训练；先对检索到的候选实体去噪，再纠错 | CommonVoice、STOP 上 WER 平均相对降 28%；检索候选的精度只有 0.139 | HTML 正文 |
| [Failing Forward / DARAG](https://arxiv.org/abs/2410.13198) | 2024-10（ACL Findings 2025） | 用 LLM 加 TTS 造合成纠错数据，再加实体检索 | 域内降 8–30%，域外降 10–33%；文本侧纠错很难泛化到没见过的错误类型 | 摘要 |
| [LLM GER for Rare Words](https://arxiv.org/abs/2505.17410) | 2025-05 | 用含稀有词的合成文本微调，输入 N-best 和音素 | 抑制了纯文本纠错的过度纠正；数字未核实 | 摘要 |

### 说字、中文拼写纠错与字形知识

| 工作 | 日期 | 要点 | 读到 |
| --- | --- | --- | --- |
| [Spoken Name Capture](https://arxiv.org/abs/2203.15833) | 2022-03 | 呼叫中心“边说边拼”姓名，在 ASR 输出后接 Seq2Seq 还原；是英文、文本侧的做法 | 摘要 |
| [Pinyin Regularization](https://arxiv.org/abs/2407.01909) | 2024-07（Interspeech） | ChineseHP 72.4 万对；在提示或微调中加入假设文本的拼音，稳定提升 LLM 纠错 | 摘要 |
| [LLM Should Understand Pinyin](https://arxiv.org/abs/2409.13262)（华为） | 2024-09 | 输入 1-best 的拼音，做拼音与文字的多任务对齐；只用合成错误训练，在 Aishell-1 和 Common Voice 上优于纯文本输入 | 摘要 |
| [SpellGCN](https://arxiv.org/abs/2004.14166)、[ReaLiSe](https://arxiv.org/abs/2105.12306)、[PLOME](https://aclanthology.org/2021.acl-long.233/) | 2020–2021 | 文本侧中文拼写纠错，把音近、形近知识注入 BERT | 摘要 |
| [RCSCB / PESC](https://aclanthology.org/2024.lrec-main.1440/) | 2024-05 | 58 万条的基准；通用指令 LLM 表现很差，专门的拼音增强模型最好 | 摘要 |
| [Visual Information in Chinese Characters](https://arxiv.org/abs/2410.09013) | 2024-10（NAACL 2025） | LLM 的部首、结构、笔画知识有限 | 摘要 |
| [Phun-Bench](https://arxiv.org/abs/2606.07300) | 2026-06（ACL 2026） | LLM 能背出读音，但不会灵活运用同音、近音 | 摘要 |
| [CNM-BERT](https://arxiv.org/abs/2608.05167) | 2026 | 把 IDS 结构树注入 BERT，长尾字的结构准确率 +9.8；摘要页的日期与编号对不上 | 摘要 |

中文说字、英文 “S as in Sam” 都没有找到公开的语音数据集或评测集。

### 合成数据到真实录音

| 工作 | 日期 | 要点 | 读到 |
| --- | --- | --- | --- |
| [How to Leverage Synthetic Speech for LLM-Based ASR](https://arxiv.org/abs/2606.29031) | 2026-06 | 合成与真实语音的差别主要在 LLM 早中层；给合成音频加 RIR 混响，再混入 25% 真实语音（13.6 小时），就能追平全用真实语音的效果 | 摘要 |
| [Beyond Oversmoothing](https://arxiv.org/abs/2410.12279) | 2024-10 | 合成与真实的 WER 比最好也有 1.46；增加 TTS 数据量和说话人数，收益递减 | 摘要 |

### 对话上下文纠错（2606.13464 及同类）

- **[2606.13464](https://arxiv.org/abs/2606.13464)**，Ontology Memory-Augmented ASR Correction for Long Text-Speech Interleaved Conversations，哈工大（深圳），2026-06，读了正文：
  - 做法：Qwen2-Audio 先识别，再用 Qwen2.5 / 3.5 文本模型纠错，免训练。从前面的人工转写里检索相关记忆，用来约束改动。
  - 数据：真实录音 MagicData-RAMC，测试集 43 个对话。
  - 结论：不加约束直接改写，多数设置比原始识别还差。记忆主要起“挑选改哪处”的作用，改动比例下降。
  - 代码：https://github.com/fangfang123gh/ontology-asr-correction
- **[MM-ContextASR](https://arxiv.org/abs/2609.19765)**，署名单位 Alibaba Token Foundry（Lei Xie 是作者之一；11-04 的调研写成“NPU 等”，以论文页面为准），2026-09，读了正文：
  - 数据：从 ASR 词典取约 17.5 万个长尾实体，用去声调拼音索引找同音、近音（n/l）的易混写法；LLM 生成五类历史（无关、隐式、显式、纠正、重复出错）；语音全部由 TTS 合成，共 87.3 万条。
  - 训练：在 Qwen3-Omni 和 Step-Audio-2-mini 上做 LoRA SFT，没有 RL，也没有 CoT。
  - 结果：实体召回 74.0 → 87.8；原始模型遇到无关历史会变差（74.0 → 69.6），SFT 后变好；真实会议录音 AliMeeting 上 CER 30.25 → 24.6。
  - 没测：误改率，以及合成到真实的迁移。
  - 证据都在上一轮对话里，出现在目标之前；没有“同一句里、说明在目标之后”的情况。
  - 评测集（合成，1,250 条）：https://github.com/llh666521/MM-ContextASR 。参考文献里列了 AmphionASR 的仓库。
- [MARS](https://arxiv.org/abs/2508.01166)（AAAI 2026）：按声学和文本相似度检索、挑选相关的历史，效果比固定窗口好。读到摘要。
- [ASR-EC](https://arxiv.org/abs/2412.03075)（EMNLP 2025 Industry）：中文文本侧纠错基准；直接用提示纠错没有效果，微调只对部分 LLM 有效，音频加转写一起用最好。读到摘要。
- [ContextASR-Bench](https://arxiv.org/abs/2507.05727)（Qwen 团队）：约 4 万条、30 多万个实体，含中文对话。可以作为外部评测参照。读到摘要。

## 对本项目的决定

- **GRPO 加上下文纠偏计划**（think 关），排在场景 TS 定位之后：
  - 题目来源：训练里已有的真实说明录音（2,187 条），加合成题。Emilia 评测集不在训练里。
  - 奖励：目标字加权；“上下文指向常用写法”的题当负例，多改要扣分。
  - 评测：纠正率与误改率一起报。
- **不做**：用合成数据追真实录音的隐式题。
- **暂缓**：注入字形知识。
- **可选**：用 MM-ContextASR Bench 和 ContextASR-Bench 做跨轮上下文能力的外部参照。

## 未找到或未核实

- 没有找到“think 在真实转写上变差”或“合成测试集高估真实表现”的直接文献。
- 场景 TS 的目标说话人误出字，没有找到直接相关的工作。
- HyPoradise、Yang et al. ASRU 2023、Knowledge prompt for Whisper、Wei et al. TASLP 2024、“Does your voice assistant remember?”：未打开，未核实。
- 中文说字的早期论文和专利（2005–2007 年、US6163767）：只看到搜索摘要。

## 大文件

本次没有下载或生成需要备份的大文件，所以没有 COS 备份。
