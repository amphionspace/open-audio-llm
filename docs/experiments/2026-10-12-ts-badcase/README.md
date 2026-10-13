# 目标说话人（TS-ASR）业务 bad case 分析（2026-10-12）

**结论：已发布的 checkpoint-34479 的 TS-ASR 对注册音频质量很敏感，业务反馈的 7 个包大部分是模型行为，不是链路问题。** 没有训练，只用已发布权重做回放和对照。

对象：发布包 `AmphionASR-1.7B/releases/20260926/amphionasr-1.7b-v3-ckpt34479`（对应 `docs/models/ckpt34479-evaluation.md`）。推理：vLLM 0.18.0，`serve.sh tsasr`，bf16，贪心，注册 3 秒加混合音频，`[SEP]` 开启。

## 一、业务反馈的 7 个包

来源：测试人员在办公室注册一个人的声纹，用 pad 播放干扰声（歌曲、对白）。数据登记为 audio-data-contract `business_feedback@ts-20261010`，评测集 `open-audio-llm/business-feedback-ts@20261010`。

- 7 个包共用同一段注册音频；回放 8 段输入，输出与反馈中的原始输出**逐字相同**，所以结果可复现。
- `model-input.wav` 与「注册 3 秒 + 源音频」逐样本一致，prompt 与 `TS_CONCAT_SYSTEM` 一致，拼接和请求格式没有问题。
- 注册音频：RMS −33 dB，3 秒里只有 44% 的帧有声（约 1.3 秒有效语音）。出错的 6 段源音频比注册音频轻 8–15 dB。

| 用例 | 现象 | 对照结果 | 判断 |
| --- | --- | --- | --- |
| case-01 | 你你你 | 源音频仅 0.45 秒；换成别人的注册则输出为空 | 过短片段幻觉，属分段问题 |
| case-02 / 05 / 06 / 07 | 输出 pad 的声音 | 换成别人的注册，全部 8 段输入中 7 段为空，只有 case-06 仍输出歌词；注册换成静音则输出更完整的歌词；源音频放大到同电平输出更多文字，不变少 | pad 声音被当作目标，电平不是原因 |
| case-03 | 复读到 512 token，`generation incomplete` | 注册换成静音同样复读（「我我我…」） | 与说话人无关的解码循环，见 `runs/decode-loop-study-20261009` |
| case-04 part-01 / 02 | 一二三四五 ×N | 同一 `segment_id`、`trace_id`，音频内容不相关（相关系数 −0.015） | 内容大概率是测试人员本人在说话；两个包的音频不一致需要前端排查 |

pad 内容为歌曲和对白是从输出文字推断的，没有听音频。

## 二、AISHELLMix test 上的注册音频对照

每个条件 300 条，单次采样，无置信区间。baseline 复现了已发布结果（2mix CER 4.40 对 4.56；负样本拒识 75.0% / 72.3% 对 78.5% / 71.2%）。

| 条件（只改注册音频） | 2mix CER | 3mix CER | 3mix 漏转写 | 3mix 转写了别人 | 3mix 负样本拒识 |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline（前 3 秒） | 4.4 | 18.8 | 1.3% | 3.4% | 72% |
| 只有 1 秒语音 | 4.9 | 23.9 | 1.0% | 8.4% | 53% |
| 前 1.5 秒静音 | 5.4 | 25.4 | 5.0% | 4.6% | 71% |
| 低通 4 kHz | 9.4 | 33.2 | 9.7% | 6.3% | 73% |
| 白噪声 10 dB | 7.5 | 34.5 | 5.3% | 12.0% | 60% |
| 混入另一个人的声音（0 dB） | **38.4** | **68.3** | **25%** | **33%** | 55% |

- 目标缺席时的误输出：2mix 负样本 75 条里 73 条（97%）、3mix 负样本 83 条里 64 条（77%）是在转写混合音频里真实说话人的内容。
- 单个干扰人、无混合的负样本（300 条）：拒识 82.0%，误输出 18.0%；源音频降 12 dB 后拒识 80.7%。
- 干净单人：CER 3.31，漏转写 0.3%；源音频降 12 dB 后 CER 3.55。
- 标准测试集的注册音频中位数 2.7 秒，63% 不足 3 秒，原本就靠补零，所以补零本身不是主要问题；训练集注册源音频中位数 11 秒且全部不短于 6 秒，训练固定取前 3 秒（`src/open_audio_llm/data/qwen3_asr.py` 的 `prepare_ts_clips`）。

## 三、训练数据里的相关事实

- AISHELLMix 训练负样本 97,830 条，对应 2spk 652,200 + 3spk 978,300，约 5.7%；负样本都是「两三个别人混合 + 缺席的注册人」，没有单个干扰人形态。
- 数据里有 `train_1spk`（326,176 条）但三阶段训练的数据清单里没有。
- 阶段 3 的 Emilia2 对话 TS 数据负样本比例较高（35,456 对 45,652 + 17,978）。
- 未合并的远端分支 `origin/docs/ckpt34479-turn-taking-eval` 记录了真实会议话轮 TS 结果：AliMeeting 2mix / 3mix CER 7.28 / 28.13，AISHELL-4 17.89 / 25.94，AMI-SDM WER 17.83 / 33.31；负样本拒识 86.5% / 88.4% / 77.2%。当前文档里没有这一节。

## 四、局限

- 业务反馈只有 7 个包，同一注册音频、同一测试人员、办公室环境，与客户的手机原始麦克风、10–50cm、户外场景不一致。
- 噪声和低通是人工合成，污染用的是随机另一个说话人；全部对照在 AISHELLMix 合成测试集上，没有在真实会议上验证。
- 「注册音频质量差是主因」是与上面对照一致的推断，没有对这批真实注册音频单独验证。
- 超过 3 秒的注册长度没有测：服务端按 3 秒切分，模型也只训练过 3 秒。

## 五、后续方向（需要确认，未执行）

1. 不需要训练：前端过滤过短片段；注册音频质量检查（有效语音少于约 2 秒、电平过低或底噪过大时提示重录）；复读保护；排查 case-04 同一 segment 出现不同音频。
2. 训练侧：注册音频随机长度（1–8 秒）、随机起点，加入污染、窄带、噪声增强；增加「单个干扰人」负样本；加入真实麦克风负样本；加入 `train_1spk`。
3. 业务反馈数据作为回归集持续登记（`business_feedback`，每批一个版本），每次训练后复测。

## 接入评测

- 数据：audio-data-contract 分支 `feat/business-feedback-ts`，`business_feedback@ts-20261010`（`target_speaker` 4 条期望输出为空，`pending_labels` 4 条参考文本未确认、不打分）。登记脚本 `scripts/business_feedback/ingest.py`，以后每个反馈批次登记为新版本。
- 评测集：AmphionEval 分支 `feat/business-feedback-ts-eval-set`，`open-audio-llm/business-feedback-ts@20261010`，用 `ae open-audio-llm ts-asr` 评测（TS 服务，`tsasr-server-url`）。
- 核验：用已发布 34479 跑通，`false_alarm_rate = 1.0`（4 条目标缺席全部误输出）。批量请求下 case-07 的输出是「你是否还能面带微笑？」，与反馈原始输出不同，原始输出保存在记录的 `metadata.reported_output`；vLLM 批处理下输出有波动。

## 复现

- `probes/probe_ts.py`、`probe_neg1.py`、`probe_cases.py`、`analyze.py`：向本地 vLLM 服务（`http://127.0.0.1:8011`）发请求；`probe_ts.py` 和 `probe_cases.py` 通过脚本同目录下的 `rel/`（发布包的本地拷贝）读取 `runtime/ts_prompt.py`。
- 原始输出：`results/enrollment-probe.jsonl.gz`（第二节全部条件）、`results/single-interferer.jsonl.gz`、`results/badcase-replay.jsonl`。
- 评分：`python probes/analyze.py <jsonl>`，归一化为去标点后的字符编辑距离。
