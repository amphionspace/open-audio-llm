# 上下文纠偏评测集 v1 与基线（2026-10-18）

**结论：34479 和官方 Qwen3-ASR-1.7B 基本不会按同一段话里的上下文改写法。** 说话人已经说明了"灰是灰色的灰"，模型仍写常用字。合成题的纠正成功率是 8.1%（34479）和 7.3%（Qwen3-ASR）；即使说明本身已经听清（拼音对得上），纠正率也只有约 8%。所以问题不在"没听清"，而在"听清了没用上"。TS 模式和普通 ASR 一样（8.0%）。隐式语义题两种写法都对的只有约 7%，说明模型几乎只按先验选字。34479 与同底座的官方模型基本持平，客户反馈的问题来自底座，不是 SFT 造成的退化。

下一步（训练方案另定）：用同一个生成器造训练数据。第一轮不加 CoT 直接 SFT（`labels.cot` 已保留）。评测看纠正成功率和误改率，CER 只作守门。

## 评测集

`context_correction_zh@v1-20261018`，AmphionEval 评测集 `open-audio-llm/context-correction-zh@v1-20261018`。只用于评测，不进训练。

| split | 任务 | 条数 | 时长 | 内容 |
| --- | --- | --- | --- | --- |
| `synthetic` | asr | 1,590（795 对） | 7.2 h | LLM 出题，Qwen3-TTS 克隆测试划分说话人朗读，平均 16 s（7–29 s） |
| `synthetic_ts` | ts_asr | 1,532（766 对） | 6.7 h | 同一批题中不超过 26.5 s 的对，配同一说话人另一句真实录音作注册音频 |
| `mined_real` | asr | 127 | 0.6 h | WenetSpeech 播客真实录音里说话人自己讲写法的片段，平均 17 s |
| `quarantine` | asr | 425 | 3.1 h | 音频没过质检的条目，保留原因，不打分 |

题目按子类型成对出：

| 子类型 | 对数 | 例子 |
| --- | --- | --- |
| 组词 | 100 | 我叫王光灰，灰是灰色的灰 |
| 拆字 | 95 | 耳东陈、立早章 |
| 偏旁结构 | 100 | 单人旁两个土的佳 |
| 否定对比 | 100 | 不是光辉的辉，是灰色的灰 |
| 同名一致性 | 100 | 名字出现多次，只有一处带说明，每处都要写对 |
| 隐式语义 | 300 | 没有说明，后文语义决定前面同音词的写法（期终/期中） |

- **成对**：每道"纠正题"（上下文指向少见写法）都配一道"确认题"，场景和读音相同，上下文指向常用写法。一律保留常用字的模型确认题满分、纠正题零分；一律改成少见字的模型正好相反。两题都对才算成对正确。隐式语义两题的前半句完全相同，只按成对计分。
- 约三成显式题另含一个必须保持不变的同音常用词（保护词）。
- 说明的位置：在被测词之后占 80%，之前占 20%。距离分三档：紧挨着、隔一两句、隔 10 s 以上。
- 拆字只有 95 对：通过复核的 137 对里，两条都过音频质检的只有 95 对，按实际数收录。
- 音色：AISHELL-3 和 KeSpeech 测试划分各 30 个说话人。以后用同一生成器造训练数据时不能用这些说话人。

## 怎么造的（LLM 出内容，代码只做机械校验）

脚本在 audio-data-contract `scripts/context_correction/`，配方 `recipe.yaml`。所有 LLM 调用（prompt、模型、原始回复）存档在 `/workspace/data/datasets/context_correction_zh/source/v1-20261018/llm-log/`。

| 步骤 | 做法 | 结果 |
| --- | --- | --- |
| 出题 | qwen3.6-plus（thinking）每次按 5 个规格各写一对，用标记标出目标词、说明、保护词，给出常用写法 | 600 次调用，3,000 对 |
| 复核 | deepseek-v4-pro 独立判断：写法是否被上下文唯一确定、不看说明时是否确实写常用字、是否同音、说法是否自然、说明描述是否准确 | 1,393 对通过（46%） |
| 机械校验 | 带调拼音完全相同、两题互换写法、目标词不在标记外出现、常用写法与另一种写法一致、隐式题前半句相同、字数 | 与复核合计隔离 1,607 对（复核否决 1,316，机械校验 707，有重叠） |
| 朗读 | Qwen3-TTS-12Hz-1.7B-Base 零样本克隆，一对题用同一音色 | 2,786 条 |
| 音频质检 | SenseVoiceSmall 转写，整段无调拼音错误率 ≤ 0.10，说明部分无调拼音全对，目标词带调拼音全对，时长 ≤ 29.5 s | 2,479 条通过（89%） |
| 收录 | 每个子类型按生成顺序取前 N 对两条都通过的，不看被测模型的输出 | 795 对 |

隐式语义的复核通过率最低（33%），最常见的否决理由是"只看前半句其实并不歧义"。

音频质检拦下的主要是真问题：罕见字读错（邗）、前后鼻音混淆（程/陈、景/金）、姓氏读音与普通读音不同（姓"纪"读 jǐ，题目就不再同音）。另有少量是拼音库读错多音字（"博闻强识"的"识"）造成的误判，按规则隔离，不人工复核。

另外修过两个问题：
- TTS 偶尔失控，生成 655 s 的长音频，这类直接按时长判不通过；
- 后补的题沿用已合成对的音色，保证同一对题始终是同一个声音。

### 真实录音挖掘

| 步骤 | 做法 | 结果 |
| --- | --- | --- |
| 切窗 | 同一录音相邻分段合成 ≤ 25 s 的窗口 | 1,194,252 个窗口 |
| 扫描 | 本地 Qwen3.6-27B（vLLM 0.18，单卡 A800，约 90 窗口/s），每批 20 个窗口，只找说话人自己讲写法的 | 1,338 个命中（0.11%） |
| 标注 | 两个 LLM 各自写出按上下文修正的转写：转写一致、目标位置重叠、读音不变、去掉修正后等于原转写才收录；同一节目重复的同一句只留一条 | 1,221 个不同文本 → 231 条一致 → 去重后 212 条 |
| 质检 | 同合成部分 | 127 条通过 |

127 条里 46 条是语料原转写写错、需要按上下文改正的，81 条原转写已经是对的。没过质检的主要是语料原转写在别处也有错漏，这种参考答案本身不可靠。

两处试过后放弃的做法：
- 第一版扫描 prompt 还要求找"语义上应写成另一个字"的行，结果把转写错字全标出来了（命中 19.7%），这类改由合成的隐式语义题覆盖；
- wenetspeech_youtube 均匀抽 1 万窗口只命中 0.04%（多为影视剧），本轮没扫。

**防泄漏**：挖掘用到的 190 个源录音列在 `source/v1-20261018/excluded-recordings.txt`（`corpus:recording_id`），以后训练用 wenetspeech 时必须排除。wenetspeech_podcast 不在 34479 的训练清单里，但 replay_v3 可能含同源录音。即使训练见过，见过的也是语料原来的错误转写，只会让这部分题更难，不会更容易。

## 指标

AmphionEval `integrations/open_audio_llm/context_correction.py`，确定性打分，不用 LLM 判分。归一化与同一 summary 里的 CER 相同。参考与输出逐字对齐（rapidfuzz）。

| 指标 | 含义 |
| --- | --- |
| 纠正成功率 `correction_rate` | 纠正题：被测字（正确写法与另一种写法不同的那几个字）全对。名字里没被说明的字写错不算（刘木阳/刘牧阳） |
| 沿用率 `unchanged_rate` | 纠正题：仍写成上下文排除的那个写法 |
| 保持率 `keep_rate` | 确认题：被测字全对 |
| 误改率 `over_correction_rate` | 确认题：写成了另一种（少见）写法。写成第三种写法算保持失败，不算误改 |
| 说明听清率 `evidence_rate` | 说明部分无调拼音全对。说明里的同音字写错（"凯是木字旁"）仍算听清 |
| 听清后纠正率 `conditional_correction_rate` | 说明听清的纠正题里的纠正成功率，用来区分"没听清"和"听清了没用上" |
| 成对正确率 `pair_accuracy` / `implicit_pair_accuracy` | 一对题都对 |
| 真实录音准确率 `mined_accuracy` | 真实录音的被测词写对 |

全部指标按子类型、距离、位置、实体类型、来源和模式（asr/ts_asr）分组，并输出每条结果（`context-correction-scores.jsonl`）和解码成本：输出 token 数、截断条数、延迟。

## 基线结果

都用 `ae open-audio-llm after-training` 跑，贪心解码，`max_new_tokens=512`，没有截断。配置里 `publish.enabled: false`，经用户同意后用 `ae publish` 补发布到 COS（`anfeiweng-audiollm-resource-1373259071:amphion-eval`，记录在 `records/asr/2026-10/`）。

| 模型 | 推理 | AmphionEval 运行 / W&B |
| --- | --- | --- |
| checkpoint-34479 | 普通 ASR：交付包 `serve.sh asr`；TS：open-audio-llm main（含 #48）的 SEP 服务，关闭 prefix cache；vLLM 0.18.0 | [`run-20261019T011627Z-c082c6aa`](https://wandb.ai/1016097967-amphion/open-audio-llm/runs/run-20261019T011627Z-c082c6aa) |
| Qwen3-ASR-1.7B 官方 | 官方 vLLM 0.18.0；没有 TS 模式，只跑普通 ASR 的两个来源 | [`run-20261019T011627Z-31dd04ba`](https://wandb.ai/1016097967-amphion/open-audio-llm/runs/run-20261019T011627Z-31dd04ba) |

合成部分（百分比；34479 普通 ASR / 34479 TS / Qwen3-ASR）：

| 指标 | 34479 ASR | 34479 TS | Qwen3-ASR |
| --- | --- | --- | --- |
| 纠正成功率 | 8.1 | 8.0 | 7.3 |
| 沿用率 | 77.2 | 75.9 | 78.4 |
| 说明听清率 | 94.9 | 93.0 | 99.2 |
| 听清后纠正率 | 8.0 | 7.9 | 7.4 |
| 保持率 | 79.8 | 77.6 | 82.4 |
| 误改率 | 6.1 | 7.0 | 4.4 |
| 保护词保持率 | 87.9 | 86.9 | 88.7 |
| 成对正确率（显式 + 同名） | 2.8 | 2.7 | 3.2 |
| 隐式语义成对正确率 | 7.7 | 6.1 | 6.7 |
| CER | 2.8 | 3.3 | 2.5 |

按子类型（34479 普通 ASR 与 TS 合计，纠正成功率 / 保持率）：组词 7.7 / 86.7，拆字 8.6 / 79.5，偏旁结构 6.1 / 66.7，否定对比 7.2 / 83.5，同名一致性 10.8 / 77.4。说明在被测词之前时纠正率高一些（14.3，之后为 6.4）。

真实录音：

| | 34479 | Qwen3-ASR |
| --- | --- | --- |
| 原转写已经对（81 条）：写对 | 90.1 | 92.6 |
| 需要纠正（46 条）：写对 | 32.6 | 30.4 |
| 需要纠正（46 条）：仍写原错误写法 | 56.5 | 50.0 |

怎么读这些数：
- 确认题保持率约 80%，但真正误改成少见写法的只有 4–7%。剩下约 14% 写成了第三种写法，比如说了"木字旁加个皆字"仍写"郑凯"，说了"鸟叫的鸣"仍写"钟明"。这同样是没用上上下文，不是数据问题：说明听清率 95% 以上，抽查的失败条目说明都已听清。
- 真实录音的纠正率（约 31%）高于合成（约 8%）。这批题来自语料里原转写写错的地方，多为常见词，样本也少（46 条），不和合成部分直接比较。

此前两次运行（`run-20261019T005711Z-*`、`run-20261019T010907Z-*`）作废，原因：
- 第一次：after-training 启动评测子进程时清掉了 `PYTHONPATH`，用的是已安装的 0.8.0，没有本分支的打分；
- 第二次：目标按整个名字判分，说明逐字判分。

现在改用克隆的 conda 环境 `amphion-eval-ctxcorr`，以 editable 方式安装分支代码。两次运行的推理输出与本次一致，只是打分口径不同。

## 复现

```bash
# 数据（audio-data-contract worktree，分支 feat/context-correction-benchmark）
O=/workspace/data/datasets/context_correction_zh/source/v1-20261018
load.sh aliyun-llm -- python scripts/context_correction/generate.py text   --out $O
load.sh aliyun-llm -- python scripts/context_correction/generate.py review --out $O
python scripts/context_correction/generate.py select --out $O
python scripts/context_correction/voices.py --roots /workspace/data/registry/roots.json --out $O
CUDA_VISIBLE_DEVICES=<gpu> <qwen-tts python> scripts/context_correction/tts.py --source $O --batch 16 [--shard K/N]
CUDA_VISIBLE_DEVICES=<gpu> python scripts/context_correction/qc.py --source $O
python scripts/context_correction/mine.py windows|scan|label|extract --source $O   # scan 需本地 Qwen3.6-27B vLLM 服务
python scripts/context_correction/qc.py --source $O --items $O/mined/wenetspeech_podcast/items.jsonl \
  --clips $O/mined/wenetspeech_podcast/clips.jsonl --out $O/mined/wenetspeech_podcast/qc.jsonl
python scripts/context_correction/build.py --source $O

# 基线（服务起好后）
ae open-audio-llm after-training --handoff baselines/<model>/handoff.json \
  --config baselines/<model>/after-training-context-correction.yaml
```

服务启动命令：

```bash
# 34479 普通 ASR（交付包）
cd /workspace/model/AmphionASR-1.7B-v3-ckpt34479 && PYTHON=<vllm-serving python> PORT=18120 GPU_UTIL=0.25 bash serve.sh asr
# 34479 TS（open-audio-llm main）
PYTHONPATH=$PWD/src AMPHION_TSASR_INSERT_SEP=1 AMPHION_TSASR_ENROLL_SEC=3 AMPHION_TSASR_ENROLL_NOPAD=0 \
  COT_TSASR_ENROLL_RANDOM=0 COT_AUDIO_CHUNKED_ATTN=0 VLLM_WORKER_MULTIPROC_METHOD=spawn \
  vllm serve /workspace/model/AmphionASR-1.7B-v3-ckpt34479/model --port 18123 --served-model-name amphionasr-v3-tsasr \
  --limit-mm-per-prompt '{"audio": 1}' --gpu-memory-utilization 0.25 --max-model-len 8192 \
  --hf-overrides '{"thinker_config": {"audio_config": {"n_window_infer": 1000000000}}}' --trust-remote-code --no-enable-prefix-caching
# Qwen3-ASR-1.7B 官方
VLLM_PLUGINS="" vllm serve /workspace/model/Qwen3-ASR-1.7B --port 18121 --served-model-name Qwen/Qwen3-ASR-1.7B \
  --dtype bfloat16 --limit-mm-per-prompt '{"audio": 1}' --gpu-memory-utilization 0.25 --max-model-len 8192 --trust-remote-code
```

## 已知局限

- 真实录音只有 127 条（计划 ≥ 200），其中需要纠正的 46 条；要扩充可再扫 wenetspeech_youtube（306 万窗口，单卡约 10 h）。
- 合成题的"常用写法"由出题 LLM 判断，不一定是被测模型的先验，所以确认题失败里有一部分是第三种写法（见上）。
- 音频质检用 SenseVoiceSmall 和拼音库，个别多音字会误判；没有人工听审。
- 合成说话人只有朗读和手机录音两种风格，没有远场和噪声。
