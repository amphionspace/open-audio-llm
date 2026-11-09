# 15 普通 ASR 的 None 偏置：收益不通用，不上线

## 结论

**不上线。** 在普通 ASR 服务上给 ` None` 加偏置（−1 或 −2），想把 GRPO 后整句被吞的语音拿回来。全部普通 ASR 集合合计收益极小，而且不同集合方向不一致：
- 偏置 −2：测试部分合计少错 431 个单位（参考约 340 万），CER 约降 0.01 个点；
- 客户通用 v5 的 CER 上升 11.7 个点；
- 抗噪 babble 在空参考上的误报成倍增加。

不满足"收益通用"的要求，普通 ASR 服务保持偏置为 0。

被吞的整句只有一部分能靠偏置拿回来，拿回来的同时，客户和噪声集上会多出凭空出字。这和 TS 的发现一样：调门槛只是在误报和漏识之间换。

## 结果（只统计加偏置后输出改变的句子；错误数为负表示变好）

按样本 id 的 md5 固定切分：开发 30%，测试 70%。开发和测试的方向一致（−2 的合计：开发 −125，测试 −431）。

| 测试部分 | 偏置 −1 错误数变化 | 偏置 −2 错误数变化 | 说明 |
|---|---:|---:|---|
| **全部普通 ASR 合计** | **−90** | **−431** | 参考约 340 万单位 |
| gigaspeech | +110 | −186 | −2 修好 49 句被吞的整句，但空参考上的误报从 32 升到 92 |
| CHiME-4（英文，仅参考） | −94 | −302 | 被吞的整句全部修好，−2 的 CER 约降 2.2 个点 |
| 警务场景 | −78 | −157 | |
| wenetspeech | −14 | −113 | |
| **客户通用 v5（109 条）** | +4 | **+56（CER +11.7 个点）** | 3 条输出了大量错字 |
| 川话 wenetspeech_chuan | +1 | +90 | 被吞的句子补出来了，但错字更多 |
| **抗噪 babble v2 / v3：空参考上的误报** | 3→5 / 8→14 | **2→12 / 4→20** | |
| common_voice_en | +30 | +35 | |

逐集完整表见 `bias-m1.txt` 和 `bias-m2.txt`（JSON 版本同名）。babble v3 全是空参考，正式评测的参考单位为 0，表里它的 CER 变化没有意义，只看误报变化。

## 方法

- **模型和评测**：模型是"两轮相加"（whitebox-merge-sum-full）。普通 ASR 服务挂上 `open_audio_llm.integrations.vllm.presence_bias:PresenceBias`（vLLM 0.18.0 logits processor），只在 `language` 之后的语言名 token 上给 ` None` 加 `AMPHION_PRESENCE_BIAS`。评测集与"两轮相加"原有的普通 ASR 评测完全相同，配置见 `configs-m1/`、`configs-m2/`。对照（偏置 0）直接用"两轮相加"已有的结果，后端、版本、数据都相同。
- **打分**：`scripts/presence_bias_compare.py` 只对输出发生变化的句子重新打分，用 AmphionEval 的 `tokenize_text` 加 token 级编辑距离，口径与 AmphionEval v3 相同。CER 变化 = 错误数变化 ÷（该集参考单位 × 所在部分占比）。
- **代码**：处理器在分支 `feat/presence-decision-bias`（dc40d75），没有合入 main，不进入部署。

## 记录

- 评测原始输出：`whai:open-audio-llm/runs/whitebox-20261107/tasks/15-presence-bias/ae-runs/`（偏置 −1、−2 各一个目录）。W&B 记录在 AmphionEval 各次 run 中，名称前缀为 `amphionasr2-grpo-whitebox-sum-full-presencebias-m1/m2`。
