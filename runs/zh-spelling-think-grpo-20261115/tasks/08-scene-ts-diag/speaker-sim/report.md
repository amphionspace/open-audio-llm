# 合成 train_ts 的 enrollment-mixture 说话人相似度（ECAPA 余弦）

## 结论
假设不成立。合成 `train_ts` 的 enrollment 与 mixture 相似度（均值 0.668，中位 0.694）比真实 TS 训练数据（0.54–0.63）和 scene-ts 目标在场项（0.49）都更高，不低。只有 9.5% 的合成条目低于 0.5；scene-ts 里“说话人不是目标”的条目均值 0.22、98.6% 低于 0.5，与合成数据几乎不重叠（合成数据的 p10 为 0.505，已高于“错人”组的 p75 0.29 以上）。所以没有证据表明“真录音 enrollment + 克隆声音 mixture”教会了模型接受低相似度的人当目标。

## 结果表（ECAPA 余弦，enrollment vs 整段 mixture）
| 组 | n | 均值 | 中位 | p10 | p25 | p75 | <0.4 | <0.5 | enroll 中位时长(s) | mix 中位时长(s) |
|---|---|---|---|---|---|---|---|---|---|---|
| **合成 context_correction_synth_zh@train-v1-20261114 train_ts** | 2917 | 0.668 | 0.694 | 0.505 | 0.590 | 0.759 | 2.8% | 9.5% | 5.2 | 15.9 |
| 真实 ts_real_train alimeeting | 500 | 0.474 | 0.497 | 0.180 | 0.280 | 0.669 | 39.2% | 51.0% | 4.5 | 17.7 |
| 真实 ts_real_train aishell4 | 500 | 0.515 | 0.553 | 0.235 | 0.353 | 0.673 | 32.0% | 41.6% | 6.0 | 18.1 |
| 真实 ts_real_train ramc | 500 | 0.626 | 0.689 | 0.318 | 0.524 | 0.775 | 15.8% | 23.6% | 4.6 | 16.5 |
| 真实 ts_real_train 三者合计 | 1500 | 0.538 | 0.586 | 0.226 | 0.357 | 0.719 | 29.0% | 38.7% | | |
| 真实 ts_scene_pool zh alt-twins（单句干净） | 500 | 0.621 | 0.634 | 0.459 | 0.546 | 0.706 | 5.6% | 14.8% | 3.7 | 2.5 |
| scene-ts 分开讲话 目标在场 | 884 | 0.488 | 0.506 | 0.211 | 0.353 | 0.639 | 32.2% | 49.4% | 5.7 | 21.9 |
| scene-ts 重叠 目标在场 | 946 | 0.494 | 0.514 | 0.186 | 0.328 | 0.674 | 35.0% | 48.2% | 5.7 | 15.0 |
| scene-ts 分开讲话 目标缺席（错人） | 16 | 0.264 | 0.233 | 0.174 | 0.182 | 0.325 | 87.5% | 100% | 5.6 | 14.8 |
| scene-ts 重叠 目标缺席（错人） | 54 | 0.203 | 0.207 | 0.044 | 0.124 | 0.283 | 92.6% | 98.1% | 5.4 | 13.0 |
| scene-ts 目标在场合计 | 1830 | 0.491 | 0.511 | 0.202 | 0.341 | 0.651 | 33.7% | 48.8% | | |
| scene-ts 目标缺席合计 | 70 | 0.216 | 0.208 | 0.065 | 0.144 | 0.289 | 91.4% | 98.6% | | |

逐条结果：同目录 `all_items.csv`（全部）和 `sim_<组名>.csv`。合成 train_ts 全部 2917 条均为非空 target，无读取失败（errors 0）。

## 对比解读
- 合成（0.668）与最干净的真实对照 ts_scene_pool（单句无干扰，0.621）和 ramc（0.626）接近，略高。合成偏高符合预期：TTS 克隆的 mixture 是干净单人语音，没有噪声和他人说话。
- 真实 alimeeting/aishell4 与 scene-ts 目标在场项的相似度偏低（约 0.47–0.52），但原因主要在 mixture 端（见限制），不是“目标不像”。
- 合成数据的低尾（<0.5 占 9.5%）仍比 scene_pool 的 14.8% 少；只有 2.8% 低于 0.4，而“错人”组 91% 低于 0.4。克隆失真造成的“像错人”的样本在合成集中很少。
- 反过来看：真实训练数据（alimeeting/aishell4）和 scene-ts 目标在场项里本来就有约三分之一低于 0.4，说明在这些场景里“目标”的相似度本就很低；这不能由合成数据解释。

## 限制
1. 真实训练数据的 mixture 并非单一说话人：ts_real_train 的 mixture 是会议录音里约 17–18 秒的窗口（远场 8 通道取 channel 0，含他人说话、混响），enrollment 是同一说话人另一时间段。scene-ts 的 mixture 含噪声、对方说话人（分开讲话 1 个干扰人，重叠同时说话）。按要求一律拿 enrollment 与整段 mixture 比，所以这两类相似度被他人声音、噪声稀释而偏低，与合成的干净单人 mixture 不是同口径；它们说明“真实场景的目标相似度可以多低”，而不是“克隆比真人更像”。
2. 合成 mixture 是整段 TTS（约 16 秒），没有加噪声或干扰；训练时是否再在线混入干扰由数据配置决定，本测量只看原始 slot 音频，未模拟训练时的在线混合。
3. 本次真实训练数据取的是 `v1-20261017-utts-punc-20261102` 的 ts_real_train（alimeeting/aishell4/ramc，各 500 条，目标非空，按固定随机种子 20261116 抽样）和 `alt-twins-20261024-punchw-20261102` 的 ts_scene_pool zh（500 条）。注意 ts_real_train 记录引用的是上游 `v1-20261017-utts` 音频索引，已按引用解析。
4. scene-ts 数据实为 `police_scene_bench@v1-20261028`（评测 run 的 samples.jsonl，共 1900 条，含 dev-/far- 场景）；ae 里 `tdtech/scene-ts@v1-20261028` 对应该数据集，但我没有单独再核对 eval set 的成员列表。按 id 含 `-sep-` 分组，按 `reference` 是否为空分目标在场/缺席。目标缺席仅 70 条（分开 16、重叠 54），尤其分开讲话组 n=16，统计量不稳。
5. 样本不等量（合成全量，真实抽样），且 enrollment 较短（中位 4–6 秒），ECAPA 对短片段有噪声；只测了“相似度”，没有测相似度与模型实际误转写的关系。
6. 这只是否定了“合成数据平均相似度低”这一条机制；模型在轮流说话场景转写他人的原因另需定位（如合成 mixture 全是目标说话、缺少“目标缺席/换人”样本，是否由 absent 比例引起，本次未测）。

## 运行的命令
```
D=/workspace/workspace/mingdong/.claude/tmp/claude-1001/-workspace-workspace-mingdong-repo-open-audio-llm/2af4df3f-ad0f-43cf-8d8e-a2c62a1ab3a3/scratchpad/ts-sim
NPROC=32 /workspace/workspace/mingdong/miniconda3/envs/k2/bin/python -I $D/ts_sim.py $D     # 约 2.5 分钟，CPU
/workspace/workspace/mingdong/miniconda3/envs/k2/bin/python -I $D/summarize.py | tee $D/summary.md
```
（oal-train 环境没有 onnxruntime，改用 k2 环境；`ts_sim.py` 中 fbank/ONNX 前处理与 `adc-ctx-emilia/scripts/ts_benchmark/speaker_embeddings.py` 的 `make_embedder` 逐行一致：16 kHz、80 维 kaldi fbank、hamming、dither 0、乘 32768、减均值，输出 L2 归一化。）
