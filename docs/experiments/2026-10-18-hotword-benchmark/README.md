# 热词评测集 v1 与基线（2026-10-18）

**结论：**

- **评测集已发布。** 两个基线全量跑完，各 264,681 个请求，失败 0。中文 `k50` 已在 [v2](../2026-10-19-hotword-bench-v2/README.md) 换成热词池检索列表；v2 只比 v1 难一点，新的评测用 v2。
- **中文：给列表能涨召回，但 34479 误触发偏多。**
  - real-zh 带正确词的 50 词列表（`k50`）：召回 34479 81.9%、Qwen3-ASR-1.7B 89.4%，误触发率 20.3% 和 8.4%。
  - 列表里没有正确词（`k50_no_gold`）时，34479 有 57.6% 的句子写出了列表外的词，ContextASR 上是 95.4%；Qwen 是 19.8% 和 61.8%。
  - 误触发几乎都是读音相近的易混词。
- **英文：34479 在非语音片段上会编文字，给列表后更严重。**
  - GigaSpeech 有 5,443 条片段参考文本为空（音乐、噪声等）。34479 在这些片段上不给列表就插入 3.9 万个词，给列表后涨到 9–12 万，常把列表词串成句子，直到 512 token 上限。
  - 所以 GigaSpeech 给列表后 WER 从 20.9% 涨到 33.7–41.3%。去掉这些片段，WER 只是 11.2% → 11.6–11.9%，Qwen 是 10.3% → 10.2–10.6%。
- **推理栈与论文一致。** AISHELL1-NE 官方 400 词列表下，Qwen3-ASR-1.7B 召回 67.6% → 73.5%、CER 4.3% → 3.9%，与 HearInContext 报告的 67.48 → 73.09、4.25 → 3.80 一致。
- **34479 在 AISHELL 上的高分主要来自训练文本重合。** AISHELL1-NE 只有 37 句文本未出现在训练中，这 37 句上的召回是 `none` 70.5%、`official` 84.1%；全集分别是 94.8% 和 95.9%。

## 为什么要换评测集

原评测集 `open-audio-llm/qwen3-asr-hotwords@20261001` 已经测不出差别，原因有三：

- **已饱和**：34479 在 AISHELL 上的召回是 99.85%，误触发约 0.01%。
- **场景单一**：只有 CV-en 和 AISHELL 两个来源，全是朗读短句。
- **干扰词太容易**：每条只配 10 个同 split 的随机词。实际部署时列表是 RAG 检索出的 50 个相似词，训练时列表长 10–53 个词，另有 50 个易混词。

另外，原评测集缺两种测法：

- 列表里没有正确词时会不会误触发；
- 列表变长后效果怎么变。

这两种情况在部署中都会出现。

## 调研结论

| 公开集 | 规模 | 为什么用 |
| --- | --- | --- |
| Test/Dev-AISHELL1-NE（SeACo-Paraformer） | 808 / 1,334 句，官方词表 400 / 600 词 | 中文热词的公认标准；SeACo、Fun-ASR-Nano 有公开数字，Qwen3-ASR-1.7B 的公开数字来自 HearInContext |
| ContextASR-Bench Speech-ZH | 15,498 条、198 h，TTS 合成，多领域 | LLM-ASR 时代常用；`gold` 条件就是论文里的 fine context |
| GigaSpeech（按 SALMONN-2 协议重建） | 25,373 条、40 h，播客和 YouTube | 英文真实语音；L=20/50/100 可对照 SALMONN-2 和 MOSS-Audio |
| LibriSpeech is21_deep_bias（Le et al.） | 2,620 / 2,939 条，官方 N=100/1000 | 英文稀有词偏置的标准列表 |

**没有采用的：**

| 数据集 | 原因 |
| --- | --- |
| ContextASR Dialogue-ZH | 平均约 158 s，超出单请求范围 |
| CB-Conformer 的 WenetSpeech 偏置集 | 从训练集 train_m 切出，与训练数据重叠 |
| HearInContext | 和已有的上下文纠偏评测集重叠 |
| Earnings | 英文金融，不是重点 |

**本地已有热词标注的数据（对照飞书「ASR 数据统计汇总」核过）：**

- 已收录：CV 中文、AISHELL-2、AISHELL-3、KeSpeech、CV 英文这五个测试集，直接沿用已有标注。标注由 AmphionData `amphion-data hotwords` 产出，模型是 doubao-seed-2-0-lite。
- 未收录：GigaSpeech、WenetSpeech、MagicData 等只有训练集带热词，测试集没有，不能直接用。

**测法**：同一条录音在多种列表条件下各跑一次。各条件的列表都事先固定写进数据，评测时不再随机抽。

| 条件 | 列表内容 | 测什么 |
| --- | --- | --- |
| `none` | 不给列表 | 起点 |
| `gold` | 只给本句的正确词 | 上限 |
| `k50`（主条件） | 正确词 + 易混词 + 同 split 随机词，共 50 个 | 长度和部署时的 RAG Top-50 一致 |
| `k50_no_gold` | 同一批干扰词，去掉正确词 | 说的词不在库里时的误触发，部署里最危险 |
| `official` / `n100` / `n1000` / `l20` / `l50` / `l100` | 论文原表 | 和公开数字对照 |

- **易混词**：读音相同或相近、但其实是另一个条目的词，例如刘燕 → 柳岩、刘艳。
- **指标**：
  - CER/WER；
  - B-/U-CER：正确词所在位置的错误率和其他位置的错误率，按 Le et al. 官方 `score.py` 的口径算；
  - 对齐式召回、精确率、F1：参照 CB-Conformer；
  - 误触发率：有列表外词被写出的句子比例，另外按易混词 / 随机词分开统计。

## 评测集

数据 `hotword_bench@v1-20261018`（audio-data-contract），评测集 `open-audio-llm/hotword-bench@v1-20261018`（AmphionEval），评测器 `hotwords`。只用于评测，不进训练。

| 切片 | 条数 | 时长 | 正确词 | 与训练文本无重合 | 列表条件 |
| --- | --- | --- | --- | --- | --- |
| aishell1-ne（Test-AISHELL1-NE） | 808 | 1.3 h | 898 | 37 | gold、k50、k50_no_gold、official（400 词） |
| contextasr-speech-zh | 15,498 | 197.6 h | 97,673 | 15,498 | gold、k50、k50_no_gold |
| real-zh（SpeechIO 27 子集 + WenetSpeech test_net / test_meeting） | 5,487 | 9.4 h | 7,264 | 5,479 | gold、k50、k50_no_gold |
| cv-zh | 2,000 | 3.4 h | 3,426 | 1,998 | gold、k50、k50_no_gold |
| aishell2 | 1,169 | 1.0 h | 1,285 | 1,157 | gold、k50、k50_no_gold |
| aishell3 | 2,000 | 1.8 h | 2,389 | 452 | gold、k50、k50_no_gold |
| kespeech（方言口音） | 2,000 | 3.3 h | 2,561 | 1,937 | gold、k50、k50_no_gold |
| gigaspeech | 25,373 | 40.1 h | 10,072 | 24,140 | l20、l50、l100（6,807 条有正确词的另有 gold、l50_no_gold） |
| librispeech-clean / other | 2,620 / 2,939 | 5.4 / 5.3 h | 5,692 / 5,248 | 2,485 / 2,918 | n100、n1000（有正确词的另有 gold、n100_no_gold） |
| cv-en | 2,000 | 3.6 h | 2,946 | 1,989 | gold、k50、k50_no_gold |
| aishell1-ne-dev（验证组，用于挑选） | 1,334 | 1.9 h | 1,447 | 18 | 同 aishell1-ne |

另有 `quarantine` 1,185 条，不打分。其中：

- 903 条是复核模型否决了全部抽出的热词；
- 36 条是上游质检未通过；
- 其余是 GigaSpeech 里不到 0.1 s 的空片段，以及 3 条超过 120 s 的录音。

**训练文本重合**：用规范化后的文本，和 AISHELL-1/2/3、KeSpeech、CV、GigaSpeech XL、WenetSpeech 训练集以及 hw_sft / hard_hw_sft 做完全匹配。

- AISHELL1-NE test 有 771/808 句与训练文本一字不差，绝大多数来自 AISHELL-2 训练转写（音频不同，文本相同）。
- AISHELL-3 有 1,548/2,000 句同样如此。

这些条目没有删掉，而是在 `labels.train_text_overlap` 里标出来源，打分时自动分出"无重合"子集。用过 AISHELL-2 的模型（含 34479）在 AISHELL 系列上的分数会偏高，主要看 real-zh、ContextASR、CV、KeSpeech、GigaSpeech。

## 怎么造的

脚本在 audio-data-contract `scripts/hotword_bench/`，配方是 `recipe.yaml`。所有 LLM 调用（prompt、模型、原始回复）存档在 `/workspace/data/datasets/hotword_bench/source/v1-20261018/llm-log/`。

| 步骤 | 做法 | 结果 |
| --- | --- | --- |
| 真实口语热词 | doubao-seed-2-0-lite 按 AmphionData `EXTRACT_HOTWORD` 抽词（与本地热词集同一模型、同一 prompt），deepseek-v4-pro 独立复核；每个子集按 sha256(id) 顺序扫描，直到有 200 条正例或扫完 | 扫描 32,195 条，复核保留 86% 的词；共 5,487 条，4 个小子集只有 107–136 条 |
| 易混词 | doubao-seed-2-1-pro 为每个中文正确词写最多 3 个同音或近音的另一条目；deepseek-v4-pro 复核两点：写成它一定算错（同义写法、异体字、简称都不算）、确实容易混淆 | 75,800 个词、189,151 对，保留 80% |
| 列表 | 正确词 + 易混词 + 同 split 热词池随机词，补满 50 个；在规范化后出现在参考文本里的词，不能当干扰词；列表顺序用 sha256 固定打乱 | — |
| GigaSpeech | XL 训练文本词频前 K 之外的词算稀有词。K 取 20000，比 5000 更接近 SALMONN-2 的统计 | 含稀有词的句子 26.6%，平均 1.48 个（SALMONN-2 为 37%、1.5 个） |

**LLM 选型**：阿里百炼账号欠费停用，改用火山方舟，并和本地 Qwen3.8-27B 做了对比（`source/v1-20261018/model-comparison/`）。

- 易混词生成：40 个词由两个裁判复核，每词合格数如下。

  | 模型 | 每词合格数 |
  | --- | --- |
  | doubao-seed-2-1-pro | 2.05 |
  | deepseek-v4-1-flash | 1.95 |
  | deepseek-v4-pro | 1.73 |
  | glm-5-2 | 0.93 |
  | Qwen3.8-27B | 0.30（多数词不生成） |

- 热词抽取：各家差别不大，复核后的结果和 doubao-seed-2-0-lite 的 F1 在 0.88–0.94。
- 托管模型会拒答少量敏感新闻内容。拒答的批次会拆成两半重试，最后剩下的单条记为"拒答"，进 quarantine。

## 基线结果

两个模型都在部署推理栈上评测：34479 用交付包的 vLLM 服务；Qwen3-ASR-1.7B 用官方 vLLM 0.18.0，bf16。`max_new_tokens` 512，seed 42。

W&B：
- 34479：[`run-20261019T070836Z-25a03f54`](https://wandb.ai/1016097967-amphion/open-audio-llm/runs/run-20261019T070836Z-25a03f54)。中途服务被另一会话误停两次，用 `--resume` 补跑失败的请求，最终失败 0。
- Qwen3-ASR-1.7B：[`run-20261019T070836Z-94b256b7`](https://wandb.ai/1016097967-amphion/open-audio-llm/runs/run-20261019T070836Z-94b256b7)。

**中文**（%；召回 / B-CER / 误触发率）：

| 切片 | 模型 | `none` CER / B-CER | `gold` | `k50` | `k50_no_gold` B-CER / 误触发 |
|---|---|---|---|---|---|
| real-zh | 34479 | 4.7 / 9.8 | 96.8 / 2.3 / 0 | 81.9 / 7.1 / 20.3 | 18.7 / 57.6 |
| real-zh | Qwen3-ASR-1.7B | 3.8 / 8.0 | 96.3 / 2.0 / 0 | 89.4 / 4.5 / 8.4 | 8.6 / 19.8 |
| contextasr-speech-zh | 34479 | 4.0 / 7.0 | 97.3 / 2.3 / 0 | 80.8 / 6.4 / 64.6 | 13.3 / 95.4 |
| contextasr-speech-zh | Qwen3-ASR-1.7B | 1.0 / 2.9 | 98.8 / 0.4 / 0 | 93.3 / 1.7 / 29.0 | 4.4 / 61.8 |
| cv-zh | 34479 | 5.3 / 8.5 | 99.4 / 0.2 / 0 | 82.2 / 4.8 / 26.9 | 16.1 / 72.5 |
| cv-zh | Qwen3-ASR-1.7B | 6.7 / 10.4 | 94.3 / 2.2 / 0 | 86.8 / 4.1 / 13.2 | 8.2 / 28.0 |
| kespeech | 34479 | 9.4 / 14.9 | 96.1 / 2.1 / 0 | 73.5 / 8.3 / 28.4 | 20.2 / 66.3 |
| kespeech | Qwen3-ASR-1.7B | 7.4 / 12.0 | 95.1 / 2.2 / 0 | 83.7 / 6.1 / 11.3 | 10.6 / 24.3 |
| aishell1-ne | 34479 | 0.9 / 2.3 | 99.4 / 0.3 / 0 | 85.1 / 5.1 / 15.3 | 16.3 / 48.6 |
| aishell1-ne | Qwen3-ASR-1.7B | 4.3 / 14.7 | 92.7 / 3.3 / 0 | 78.6 / 8.2 / 13.6 | 14.1 / 27.5 |
| aishell2 | 34479 | 3.7 / 8.4 | 98.5 / 0.7 / 0 | 80.1 / 7.5 / 20.6 | 23.2 / 65.6 |
| aishell2 | Qwen3-ASR-1.7B | 4.3 / 10.4 | 94.3 / 2.7 / 0 | 81.2 / 7.6 / 13.3 | 12.3 / 23.7 |
| aishell3 | 34479 | 1.4 / 2.6 | 99.7 / 0.1 / 0 | 83.8 / 4.8 / 18.6 | 16.9 / 60.8 |
| aishell3 | Qwen3-ASR-1.7B | 3.3 / 6.8 | 97.5 / 1.0 / 0 | 88.6 / 3.9 / 10.3 | 7.9 / 23.1 |

**AISHELL1-NE 官方列表与训练文本未重合的子集**（召回 %）：

| 模型 | 全集 `none` | 全集 `official` | 未重合 37 句 `none` | 未重合 37 句 `official` |
|---|---|---|---|---|
| 34479 | 94.8 | 95.9 | 70.5 | 84.1 |
| Qwen3-ASR-1.7B | 67.6 | 73.5 | 45.5 | 45.5 |

AISHELL-3 未重合的 452 句：34479 的 `none` 召回 85.6%，Qwen 80.7%。

**英文**（%；WER / B-WER / 召回）：

| 切片 | 条件 | 34479 | Qwen3-ASR-1.7B |
|---|---|---|---|
| gigaspeech | `none` | 20.9 / 29.4 / 69.5 | 13.2 / 25.6 / 73.5 |
| gigaspeech | `l20` | 33.7 / 11.5 / 88.4 | 13.6 / 17.8 / 81.8 |
| gigaspeech | `l50` | 39.9 / 12.7 / 87.1 | 13.5 / 20.1 / 79.3 |
| gigaspeech | `l100` | 41.3 / 14.2 / 85.5 | 13.7 / 22.6 / 76.7 |
| librispeech-clean | `none` / `n100` / `n1000` | 1.8 / 1.3 / 2.1（WER） | 1.9 / 1.7 / 2.0（WER） |
| librispeech-clean | `n100` B-WER | 3.7 | 5.9 |
| librispeech-other | `none` / `n100` / `n1000` | 3.8 / 2.8 / 4.2（WER） | 3.7 / 3.2 / 3.8（WER） |
| librispeech-other | `n100` B-WER | 7.6 | 12.8 |
| cv-en | `k50` | 4.2 / 1.7 / 97.6 | 6.5 / 10.7 / 84.0 |
| cv-en | `k50_no_gold` 误触发率 | 6.5 | 1.6 |

- **GigaSpeech 的 B-WER**：34479 给列表后明显更低（29.4 → 11.5–14.2）。这说明它会用列表；但正如结论所说，非语音片段上的编造把整体 WER 拉高了。
- **LibriSpeech `n1000`**：两个模型都与 `none` 接近，说明 1000 词的长列表基本用不上。

## 复现

```bash
# 数据（audio-data-contract worktree，分支 feat/hotword-benchmark）
S=/workspace/data/datasets/hotword_bench/source/v1-20261018
python scripts/hotword_bench/sources.py --source $S
python scripts/hotword_bench/gigaspeech.py --source $S
load.sh volcengine-ark -- python scripts/hotword_bench/mine.py real-zh --source $S
load.sh volcengine-ark -- python scripts/hotword_bench/mine.py confusables --source $S
load.sh volcengine-ark -- python scripts/hotword_bench/mine.py confusables-review --source $S
python scripts/hotword_bench/build.py --source $S
python scripts/hotword_bench/eval_set.py > data.yaml

# 基线（服务起好后；评测环境装 AmphionEval 分支）
export AUDIO_DATA_CATALOG=/workspace/data/registry/catalog AUDIO_DATA_ROOTS_FILE=/workspace/data/registry/roots.json
ae open-audio-llm after-training --handoff baselines/<model>/handoff.json \
  --config baselines/<model>/after-training-hotword-bench.yaml
```

服务启动命令（都在一张 A800 上）：

```bash
# 34479（交付包）
cd /workspace/model/AmphionASR-1.7B-v3-ckpt34479 && PYTHON=<vllm-serving python> PORT=18120 GPU_UTIL=0.25 bash serve.sh asr
# Qwen3-ASR-1.7B 官方
VLLM_PLUGINS="" vllm serve /workspace/model/Qwen3-ASR-1.7B --port 18121 --served-model-name Qwen/Qwen3-ASR-1.7B \
  --dtype bfloat16 --limit-mm-per-prompt '{"audio": 1}' --gpu-memory-utilization 0.25 --max-model-len 8192 --trust-remote-code
```
