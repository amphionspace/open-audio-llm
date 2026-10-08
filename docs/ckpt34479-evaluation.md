# checkpoint-34479 评测结果

权重为 Qwen3-ASR-1.7B 经过三阶段全参 SFT 后的 `checkpoint-34479`。训练配置见 [三阶段 SFT](qwen3-asr-three-stage-sft.md)。模型是完整 SFT checkpoint，无需合并 LoRA。

同一套权重支持三种任务，不要混在同一个 vLLM 服务里发：

| 能力 | 怎么用 | 服务模式 |
|---|---|---|
| 普通 ASR | 一段语音，system 为空，可自动判断或指定语言 | 不插入 `[SEP]` |
| 热词偏置 ASR | 普通 ASR + system 写 `Hotwords: 词1,词2` | 同一个普通 ASR 服务 |
| 目标说话人 ASR（TS-ASR） | 注册音频前 3 秒 + 混合音频；只转写注册说话人；人不在场应输出空 | 插入 `[SEP]` |

- 中英为主。底座 Qwen3-ASR-1.7B 还列了粤语及其他语种，本轮 SFT 与本次评测只覆盖中英。
- 热词是 prompt 偏置，把词写进 system。本次评测不含双塔检索 adapter；K=0 / K=50 是随机组词协议，不是 RAG。
- TS-ASR 传输为「注册 3 秒 + 混合音频」，中间不加静音。服务分别提 Mel、分别卷积，再在 audio transformer 前插入可学习 `[SEP]`（`thinker.audio_tower.sep_token.weight`）。
- 目标说话人缺席时仍可能误输出文字，空输出不能当成可靠的说话人验证。

## Benchmark

评测后端为 vLLM 0.18.0，bfloat16，贪心解码，6 路数据并行，`max_model_len=8192`。音频编码器使用全局注意力：`n_window_infer=1000000000`。普通 ASR、热词和警务不插入 `[SEP]`；TS-ASR 设置 `AMPHION_TSASR_INSERT_SEP=1`。

中文按字（CER），英文按词（WER）；热词的 B- / U- 与主指标同一粒度。JSON 字段名统一为 `wer` / `b_wer` / `u_wer`。下表为百分比，越低越好。各测试集失败数均为 0。

### ASR-full

| 测试集 | 语种 | 指标 | n | 34479 |
|---|---|---|---:|---:|
| LibriSpeech test-clean | en | WER | 2620 | 1.66 |
| LibriSpeech test-other | en | WER | 2939 | 3.64 |
| GigaSpeech | en | WER | 19930 | 9.36 |
| MLS | en | WER | 3769 | 6.05 |
| Common Voice en | en | WER | 16402 | 8.35 |
| FLEURS en | en | WER | 647 | 5.07 |
| AISHELL-1 | zh | CER | 7176 | 0.63 |
| AISHELL-2 | zh | CER | 5000 | 2.81 |
| AISHELL-3 | zh | CER | 24773 | 1.82 |
| MagicData | zh | CER | 24279 | 1.87 |
| KeSpeech | zh | CER | 19723 | 6.97 |
| THCHS-30 | zh | CER | 2495 | 4.29 |
| Common Voice zh | zh | CER | 10647 | 5.17 |
| WenetSpeech test-net | zh | CER | 24774 | 6.44 |
| WenetSpeech test-meeting | zh | CER | 8370 | 7.41 |
| FLEURS zh | zh | CER | 945 | 3.54 |
| **合计** | | **WER/CER** | **174489** | **5.64** |

合计按全部 16 个测试集的 ref token 加权：`error / ref_tokens = 5.643%`（149666 / 2652385）。

### 热词（random，K=0 / K=50）

Common Voice 热词集。K=0：不注入热词（`Hotwords: N/A` 基线）。K=50：真实热词与从 10 万词表抽出的干扰词一并 pad 到 50，写入 `Hotwords:`。中文、英文分别用独立 100k 词表。B- 为热词覆盖片段，U- 为其余片段。

| 测试集 | K | 主指标 | B- | U- | KER | n |
|---|---:|---:|---:|---:|---:|---:|
| Common Voice zh | 0 | CER 4.32 | B-CER 8.38 | U-CER 2.62 | 22.54 | 10550 |
| Common Voice zh | 50 | CER 1.72 | B-CER 0.43 | U-CER 2.26 | 0.86 | 10550 |
| Common Voice en | 0 | WER 7.83 | B-WER 27.42 | U-WER 5.92 | 38.44 | 16294 |
| Common Voice en | 50 | WER 5.06 | B-WER 1.90 | U-WER 5.37 | 2.26 | 16294 |

### TS-ASR（concat + `[SEP]`）

正样本：LibriMix 报 WER，AISHELLMix 报 CER。负样本：注册说话人不在场，`reject` 为输出为空的比例（越高越好）。误输出比例是 `1 - reject`，不再单列。

| 测试集 | n | 指标 | 34479 |
|---|---:|---|---:|
| LibriMix test 2mix | 6000 | WER | 3.81 |
| LibriMix test 3mix | 9000 | WER | 8.63 |
| AISHELLMix test 2mix | 6000 | CER | 4.56 |
| AISHELLMix test 3mix | 9000 | CER | 17.53 |
| LibriMix test 2mix-neg | 3000 | reject | 0.910 |
| LibriMix test 3mix-neg | 3000 | reject | 0.851 |
| AISHELLMix test 2mix-neg | 3000 | reject | 0.785 |
| AISHELLMix test 3mix-neg | 3000 | reject | 0.712 |

### 话轮 TS-ASR（AliMeeting / AISHELL-4 / AMI-SDM）

这一节不是上面的 LibriMix / AISHELLMix 拼接集，也不是按官方测试语句逐条裁剪的会议语句。Amphion 是 `checkpoint-34479`：vLLM、注册前 3 秒、插入 `[SEP]`、不加静音。Cocktail 是 `Xiaomi-CocktailASR-1`，同一批 manifest 各跑了非 CoT 和 CoT。

样本从会议录音切出，规则如下：

- 只打包完整语句，不从句中切开。相邻语句间隔超过 3 秒、窗口时长超过 27 秒，或说话人会变成 4 个及以上时，先结束当前窗口。
- 留下的窗口必须正好是 2 个或 3 个说话人，窗口时长至少 1 秒。单人片段丢弃。
- 一个窗口里每个说话人一条正例。参考文本是这个人在窗内全部语句拼起来，不是单独一句。该人在窗内的语音短于 1 秒则跳过。
- 混合音频是这段窗口的远场裁剪。注册音频是窗外另一条至少 3 秒、且不与窗口重叠的语句。有近场或头戴麦时，按样本 id 哈希以远场:近场 = 3:1 选取；没有近场则用远场。
- 每个窗口一条负例：注册说话人不在该窗口的说话人集合里。
- 评测只用官方 test 场次切出的窗口，不用 train，也不用另外留出的 eval 场次。AliMeeting test 20 场，AISHELL-4 test 20 场，AMI-SDM test 16 场。AMI 另有 8 场从 train 划出的 eval，AliMeeting 的 dev 记为 eval（8 场），这两部分都没进下表。

正例按说话人展开，所以条数多于窗口数。2 人与 3 人分列。中文报 CER，英文 AMI-SDM 报 WER，越低越好。差值是 Cocktail 减去 34479，正值表示 Cocktail 错误更多。

| 测试集 | n | 指标 | 34479 | Cocktail 非 CoT | 差值 | Cocktail CoT | 差值 |
|---|---:|---|---:|---:|---:|---:|---:|
| AliMeeting 2mix | 1377 | CER | 7.28 | 51.64 | +44.36 | 52.99 | +45.71 |
| AliMeeting 3mix | 2430 | CER | 28.13 | 86.87 | +58.74 | 71.04 | +42.91 |
| AISHELL-4 2mix | 947 | CER | 17.89 | 24.32 | +6.43 | 21.50 | +3.61 |
| AISHELL-4 3mix | 1361 | CER | 25.94 | 44.32 | +18.38 | 38.94 | +13.00 |
| AMI-SDM 2mix | 553 | WER | 17.83 | 31.71 | +13.88 | 34.27 | +16.44 |
| AMI-SDM 3mix | 2556 | WER | 33.31 | 58.78 | +25.47 | 59.93 | +26.62 |

负例参考文本为空，WER 为 0 没有识别意义。`reject` 是输出为空的比例，越高越好；误输出比例是 `1 - reject`。

| 测试集 | n | 34479 reject | Cocktail 非 CoT | Cocktail CoT |
|---|---:|---:|---:|---:|
| AliMeeting neg | 1829 | 0.865 | 0.002 | 0.042 |
| AISHELL-4 neg | 1053 | 0.884 | 0.013 | 0.008 |
| AMI-SDM neg | 1465 | 0.772 | 0.006 | 0.009 |

AliMeeting 上 Cocktail 的 CER 被重复生成拉高。非 CoT 的 2mix 插入错误 33795、替换 3288；3mix 有 372 条假设退化成很长的「对对对」或「嗯嗯嗯」。CoT 只把 AliMeeting 3mix 和 AISHELL-4 明显拉低，其余集合持平或略差。Cocktail 几乎不拒识。

两边的推理前端也不相同，不能把差距全部算成切窗差异。Amphion 是注册音频前 3 秒与混合音频之间插入 `[SEP]`，中间不加静音。Cocktail 官方预处理是注册裁到 1–4 秒，再接 1 秒静音和混合音频；CoT 使用 `<think>` / `<answer>` 提示，非 CoT 使用普通转写提示。

### 警务

同一 ASR 服务上的三个测试集。`police_v6_expanded` 的 CER 和句准确率按指令文本归一化（「第四条」与「第4条」算同一句）；槽位 joint 要求 action、object、value 同时正确。术语召回是目标词在转写中的子串命中。`police_incremental_20260916` 的四类正好拆完这 10000 条，不是额外测试集。

**police_v6_expanded**（2000 条）

| 指标 | 34479 |
|---|---:|
| 归一化 CER | 0.24 |
| 句准确率 | 98.35 |
| action / object / value | 100.00 / 100.00 / 100.00 |
| 槽位 joint | 100.00 |
| 警单签收选择 joint（118） | 100.00 |
| 开关镜头摄录 joint（752） | 100.00 |
| 媒体配置 joint（474） | 100.00 |
| 预录延录 joint（181） | 100.00 |
| AI求助标记打点 joint（44） | 100.00 |
| 查询系统工具 joint（431） | 100.00 |

**police_terms_v5**（585 条，683 次术语）

| 指标 | 34479 |
|---|---:|
| 术语召回 | 98.98%（676/683） |
| 字错率 CER | 0.86 |

**police_incremental_20260916**（10000 条）

| 指标 | 34479 |
|---|---:|
| 术语召回 | 99.53%（10095/10143） |
| 行业术语 CER（4000） | 0.29 |
| 业务应用与指令 CER（2500） | 0.29 |
| 数字与号码 CER（1500） | 0.27 |
| 行业对话 CER（2000） | 0.28 |
| 整集字错率 CER | 0.33 |
