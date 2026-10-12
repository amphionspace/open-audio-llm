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

正样本：LibriMix 报 WER，AISHELLMix 报 CER。负样本：注册说话人不在场，`reject` 为输出为空的比例（越高越好），`FA` 为误输出比例。

| 测试集 | n | 指标 | 34479 |
|---|---:|---|---:|
| LibriMix test 2mix | 6000 | WER | 3.81 |
| LibriMix test 3mix | 9000 | WER | 8.63 |
| AISHELLMix test 2mix | 6000 | CER | 4.56 |
| AISHELLMix test 3mix | 9000 | CER | 17.53 |
| LibriMix test 2mix-neg | 3000 | reject / FA | 0.910 / 0.090 |
| LibriMix test 3mix-neg | 3000 | reject / FA | 0.851 / 0.149 |
| AISHELLMix test 2mix-neg | 3000 | reject / FA | 0.785 / 0.215 |
| AISHELLMix test 3mix-neg | 3000 | reject / FA | 0.712 / 0.288 |

负样本上 LibriMix 2mix / 3mix 的干扰人泄漏率（hyp 匹配干扰人文本）为 5.70% / 4.70%。AISHELLMix 负样本未统计到干扰人文本命中。

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
