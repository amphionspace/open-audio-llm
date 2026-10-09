# v2-33407 与 v3-34479 评测结果

权重对象存储位置：`/cos-amphion-delivery/AmphionASR-1.7B/releases/20260926`（v3-34479 这一份）

RAG 对象存储位置：`/cos-amphion-delivery/amphionasr-rag/20261009`

两列都是 Qwen3-ASR-1.7B 的全参 SFT checkpoint，不需要合并 LoRA。训练配置见 [三阶段 SFT](qwen3-asr-three-stage-sft.md)。

| 列名 | 阶段 | checkpoint |
|---|---|---|
| v2-33407 | 阶段 2 结束，`sft/v2-20260921-223309` | checkpoint-33407 |
| v3-34479 | 阶段 3 结束，从 33407 继续训，`sft/continue_sft_turn_taking_police/v0-20260924-205747` | checkpoint-34479 |

阶段 3 的结束步是 34479。没有 checkpoint-33479。

同一套权重支持三种任务，不要混在同一个 vLLM 服务里发：

| 能力 | 怎么用 | 服务模式 |
|---|---|---|
| 普通 ASR | 一段语音，system 为空，可自动判断或指定语言 | 不插入 `[SEP]` |
| 热词偏置 ASR | 普通 ASR + system 写 `Hotwords: 词1,词2` | 同一个普通 ASR 服务 |
| 目标说话人 ASR（TS-ASR） | 注册音频前 3 秒 + 混合音频；只转写注册说话人；人不在场应输出空 | 插入 `[SEP]` |

- 中英为主。底座 Qwen3-ASR-1.7B 还列了粤语及其他语种，本轮 SFT 与本次评测只覆盖中英。
- 热词是 prompt 偏置，把词写进 system。random 的 K=0 / K=50 从 10 万词表组词；RAG 的 K=50 用上面的 `amphionasr-rag` 从 1 万词表召回 top-50。
- TS-ASR 传输为「注册 3 秒 + 混合音频」，中间不加静音。服务分别提 Mel、分别卷积，再在 audio transformer 前插入可学习 `[SEP]`（`thinker.audio_tower.sep_token.weight`）。
- 目标说话人缺席时仍可能误输出文字，空输出不能当成可靠的说话人验证。

## Benchmark

评测后端为 vLLM 0.18.0，bfloat16，贪心解码，`max_model_len=8192`。音频编码器使用全局注意力：`n_window_infer=1000000000`。普通 ASR、热词和警务不插入 `[SEP]`；TS-ASR 设置 `AMPHION_TSASR_INSERT_SEP=1`。

中文按字（CER），英文按词（WER）；热词的 B- / U- 与主指标同一粒度。JSON 字段名统一为 `wer` / `b_wer` / `u_wer`。下表为百分比，越低越好。各测试集失败数均为 0。

v2 的普通 ASR、热词、LibriMix / AISHELLMix、话轮来自已有实验：`exp/eval_vllm/sft_v2_ckpt33407_asr_full`、`exp/eval_vllm/sft_v2_ckpt33407_hw_random`、`exp/eval/sft_v2_ckpt33407_tsasr`、`exp/eval/sft_v2_ckpt33407_tsasr_turn_taking`。v3 对应 `exp/eval_vllm/police_tt_v0_ckpt34479_asr_full`、`exp/eval_vllm/police_tt_v0_ckpt34479_hw_random`、`exp/eval/police_tt_v0_ckpt34479_tsasr`、`exp/eval/sft_v3_ckpt34479_tsasr_turn_taking`。v2 原先没有警务三项，本轮补在 `exp/eval_vllm/sft_v2_ckpt33407_police`，协议与 v3 的 `exp/eval_vllm/police_tt_v0_ckpt34479_police` 相同。RAG K=50 在 `exp/eval_vllm/sft_v2_ckpt33407_hw_rag_k50` 和 `exp/eval_vllm/sft_v3_ckpt34479_hw_rag_k50`，检索权重是 `deploy/amphionasr-rag`。

### ASR-full

| 测试集 | 语种 | 指标 | n | v2-33407 | v3-34479 |
|---|---|---|---:|---:|---:|
| LibriSpeech test-clean | en | WER | 2620 | 1.65 | 1.66 |
| LibriSpeech test-other | en | WER | 2939 | 3.81 | 3.66 |
| GigaSpeech | en | WER | 19930 | 9.40 | 9.39 |
| MLS | en | WER | 3769 | 6.10 | 6.06 |
| Common Voice en | en | WER | 16402 | 9.18 | 8.38 |
| FLEURS en | en | WER | 647 | 5.64 | 5.08 |
| AISHELL-1 | zh | CER | 7176 | 0.63 | 0.63 |
| AISHELL-2 | zh | CER | 5000 | 2.79 | 2.80 |
| AISHELL-3 | zh | CER | 24773 | 2.05 | 1.81 |
| MagicData | zh | CER | 24279 | 2.25 | 1.87 |
| KeSpeech | zh | CER | 19723 | 8.44 | 7.00 |
| THCHS-30 | zh | CER | 2495 | 4.03 | 4.30 |
| Common Voice zh | zh | CER | 10647 | 5.30 | 5.13 |
| WenetSpeech test-net | zh | CER | 24774 | 7.17 | 6.44 |
| WenetSpeech test-meeting | zh | CER | 8370 | 6.21 | 7.41 |
| FLEURS zh | zh | CER | 945 | 3.11 | 3.48 |
| **合计** | | **WER/CER** | **174489** | **5.92** | **5.65** |

合计按全部 16 个测试集的 ref token 加权。v2-33407 是 `157134 / 2652385 = 5.924%`，v3-34479 是 `149801 / 2652385 = 5.648%`。

### 热词

Common Voice 热词集。中文报 CER，英文报 WER。B- 为热词覆盖片段，U- 为其余片段。

- random K=0：不注入热词（`Hotwords: N/A`）。
- random K=50：真实热词与从 10 万词表抽出的干扰词一并 pad 到 50。
- RAG K=50：`amphionasr-rag` 做双塔召回，音频按帧与热词向量取最大相似度，从 1 万词表取 top-50 写入 `Hotwords:`。识别仍用 v2-33407 和 v3-34479。

| 测试集 | 供给 | K | n | v2-33407 | B- | U- | v3-34479 | B- | U- |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Common Voice zh | random | 0 | 10550 | 4.41 | 8.48 | 2.59 | 4.26 | 8.13 | 2.65 |
| Common Voice zh | random | 50 | 10550 | 1.94 | 0.64 | 2.51 | 1.73 | 0.47 | 2.30 |
| Common Voice zh | RAG | 50 | 10550 | 2.87 | 2.73 | 2.94 | 2.59 | 2.13 | 2.80 |
| Common Voice en | random | 0 | 16294 | 8.69 | 28.38 | 6.73 | 7.86 | 27.08 | 5.95 |
| Common Voice en | random | 50 | 16294 | 5.77 | 2.07 | 6.14 | 5.07 | 1.79 | 5.40 |
| Common Voice en | RAG | 50 | 16294 | 8.69 | 11.35 | 8.43 | 7.92 | 10.05 | 7.71 |

### TS-ASR（concat + `[SEP]`）

正样本：LibriMix 报 WER，AISHELLMix 报 CER。负样本：注册说话人不在场，`reject` 为输出为空的比例（越高越好）。误输出比例是 `1 - reject`，不再单列。

| 测试集 | n | 指标 | v2-33407 | v3-34479 |
|---|---:|---|---:|---:|
| LibriMix test 2mix | 6000 | WER | 3.87 | 3.80 |
| LibriMix test 3mix | 9000 | WER | 9.26 | 8.64 |
| AISHELLMix test 2mix | 6000 | CER | 5.01 | 4.56 |
| AISHELLMix test 3mix | 9000 | CER | 19.42 | 17.45 |
| LibriMix test 2mix-neg | 3000 | reject | 0.872 | 0.909 |
| LibriMix test 3mix-neg | 3000 | reject | 0.796 | 0.851 |
| AISHELLMix test 2mix-neg | 3000 | reject | 0.734 | 0.784 |
| AISHELLMix test 3mix-neg | 3000 | reject | 0.623 | 0.711 |

### 话轮 TS-ASR（AliMeeting / AISHELL-4 / AMI-SDM）

这一节不是上面的 LibriMix / AISHELLMix 拼接集，也不是按官方测试语句逐条裁剪的会议语句。Amphion 两列都是 vLLM、注册前 3 秒、插入 `[SEP]`、不加静音。Cocktail 是 `Xiaomi-CocktailASR-1`，同一批 manifest 各跑了非 CoT 和 CoT。

样本从会议录音切出，规则如下：

- 只打包完整语句，不从句中切开。相邻语句间隔超过 3 秒、窗口时长超过 27 秒，或说话人会变成 4 个及以上时，先结束当前窗口。
- 留下的窗口必须正好是 2 个或 3 个说话人，窗口时长至少 1 秒。单人片段丢弃。
- 一个窗口里每个说话人一条正例。参考文本是这个人在窗内全部语句拼起来，不是单独一句。该人在窗内的语音短于 1 秒则跳过。
- 混合音频是这段窗口的远场裁剪。注册音频是窗外另一条至少 3 秒、且不与窗口重叠的语句。有近场或头戴麦时，按样本 id 哈希以远场:近场 = 3:1 选取；没有近场则用远场。
- 每个窗口一条负例：注册说话人不在该窗口的说话人集合里。
- 评测只用官方 test 场次切出的窗口，不用 train，也不用另外留出的 eval 场次。AliMeeting test 20 场，AISHELL-4 test 20 场，AMI-SDM test 16 场。AMI 另有 8 场从 train 划出的 eval，AliMeeting 的 dev 记为 eval（8 场），这两部分都没进下表。

正例按说话人展开，所以条数多于窗口数。2 人与 3 人分列。中文报 CER，英文 AMI-SDM 报 WER，越低越好。差值是 Cocktail 减去 v3-34479，正值表示 Cocktail 错误更多。

| 测试集 | n | 指标 | v2-33407 | v3-34479 | Cocktail 非 CoT | 差值 | Cocktail CoT | 差值 |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| AliMeeting 2mix | 1377 | CER | 7.03 | 7.28 | 51.64 | +44.36 | 52.99 | +45.71 |
| AliMeeting 3mix | 2430 | CER | 28.69 | 28.13 | 86.87 | +58.74 | 71.04 | +42.91 |
| AISHELL-4 2mix | 947 | CER | 17.72 | 17.89 | 24.32 | +6.43 | 21.50 | +3.61 |
| AISHELL-4 3mix | 1361 | CER | 26.83 | 25.94 | 44.32 | +18.38 | 38.94 | +13.00 |
| AMI-SDM 2mix | 553 | WER | 17.36 | 17.83 | 31.71 | +13.88 | 34.27 | +16.44 |
| AMI-SDM 3mix | 2556 | WER | 39.01 | 33.31 | 58.78 | +25.47 | 59.93 | +26.62 |

负例参考文本为空，WER 为 0 没有识别意义。`reject` 是输出为空的比例，越高越好；误输出比例是 `1 - reject`。

| 测试集 | n | v2-33407 reject | v3-34479 reject | Cocktail 非 CoT | Cocktail CoT |
|---|---:|---:|---:|---:|---:|
| AliMeeting neg | 1829 | 0.847 | 0.865 | 0.002 | 0.042 |
| AISHELL-4 neg | 1053 | 0.859 | 0.884 | 0.013 | 0.008 |
| AMI-SDM neg | 1465 | 0.731 | 0.772 | 0.006 | 0.009 |

AliMeeting 上 Cocktail 的 CER 被重复生成拉高。非 CoT 的 2mix 插入错误 33795、替换 3288；3mix 有 372 条假设退化成很长的「对对对」或「嗯嗯嗯」。CoT 只把 AliMeeting 3mix 和 AISHELL-4 明显拉低，其余集合持平或略差。Cocktail 几乎不拒识。

两边的推理前端也不相同，不能把差距全部算成切窗差异。Amphion 是注册音频前 3 秒与混合音频之间插入 `[SEP]`，中间不加静音。Cocktail 官方预处理是注册裁到 1–4 秒，再接 1 秒静音和混合音频；CoT 使用 `<think>` / `<answer>` 提示，非 CoT 使用普通转写提示。

### 警务

同一 ASR 服务上的三个测试集。`police_v6_expanded` 的 CER 和句准确率按指令文本归一化（「第四条」与「第4条」算同一句）；槽位 joint 要求 action、object、value 同时正确。术语召回是目标词在转写中的子串命中。`police_incremental_20260916` 的四类正好拆完这 10000 条，不是额外测试集。

`police_v6_expanded` 用数据生成端的 `comparison_text` 和 `command_slots`：数字、型号的等价写法合并后再算字错率和句准确率，槽位比较 action、object、value。术语召回和 `police_incremental_20260916` 的 CER 先去掉标点和空格，再做子串命中和字面编辑距离。v3 的术语召回、整集 CER、槽位与已保存的评测附加统计一致。四类 CER 也按这套字面距离重算，因此和上一版文档里的 0.29 / 0.29 / 0.27 / 0.28 不是同一套数字。

**police_v6_expanded**（2000 条）

| 指标 | v2-33407 | v3-34479 |
|---|---:|---:|
| 归一化 CER | 2.93 | 0.25 |
| 句准确率 | 78.10 | 98.30 |
| action / object / value | 98.75 / 91.00 / 98.56 | 100.00 / 100.00 / 100.00 |
| 槽位 joint | 89.55 | 100.00 |
| 警单签收选择 joint（118） | 95.76 | 100.00 |
| 开关镜头摄录 joint（752） | 96.14 | 100.00 |
| 媒体配置 joint（474） | 96.20 | 100.00 |
| 预录延录 joint（181） | 25.97 | 100.00 |
| AI求助标记打点 joint（44） | 86.36 | 100.00 |
| 查询系统工具 joint（431） | 96.06 | 100.00 |

**police_terms_v5**（585 条，683 次术语）

| 指标 | v2-33407 | v3-34479 |
|---|---:|---:|
| 术语召回 | 76.72%（524/683） | 98.98%（676/683） |
| 字错率 CER | 4.63 | 0.86 |

**police_incremental_20260916**（10000 条）

| 指标 | v2-33407 | v3-34479 |
|---|---:|---:|
| 术语召回 | 95.90%（9727/10143） | 99.48%（10090/10143） |
| 行业术语 CER（4000） | 2.11 | 0.31 |
| 业务应用与指令 CER（2500） | 2.53 | 0.32 |
| 数字与号码 CER（1500） | 1.77 | 0.30 |
| 行业对话 CER（2000） | 1.75 | 0.37 |
| 整集字错率 CER | 2.06 | 0.33 |
