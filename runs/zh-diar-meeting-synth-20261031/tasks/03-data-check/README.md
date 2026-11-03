# 训练路径数据核查

- 输入：`meeting_synth_zh@v1-20261031`、[configs/synth-v1.yaml](../../configs/synth-v1.yaml)（最终配置）、arm I 的增强配置（本来源加入 `noise_exclude_datasets`）。
- 做法：[scripts/check_synth_training_data.py](../../scripts/check_synth_training_data.py)，走 `CatalogSwiftDataset`，30 / 120 / 300 s 各 150 窗、2 个 epoch，共 900 窗。
- 记录：attempt 001 只留串音电平（逐窗部分用旧配置，按计划中止）；attempt 002 用 `max_overlap: 0`，配置撤回后中止；attempt 003 是 AliMeeting + AISHELL-5、未用齐来源会议时的结果；attempt 004 用 AliMeeting 单独配置，第二轮未跑完时因配置已被替代而手动停止，无结果文件；attempt 005 是最终配置的结果。

## 最终结果（attempt 005，执行退出码 1）

配置：AliMeeting + AISHELL-5 1:1、用齐来源会议、排除 49 场问题会议，共 900 窗。

1. 泄漏通过，标注格式、时间、说话人编号全部合法，没有额外波形增强。
2. 退出码 1 来自 1 个窗口的 3 帧 100 ms 数字静音（epoch 1，index 92326）。这个窗口没有单独复现，按占比极小判定不影响训练，和 attempt 003 的情况同类。
3. 标注处有语音：行能量距窗口底噪不足 3 dB 的比例，AliMeeting 素材 1.17%（p5 5.4，p50 14.4），AISHELL-5 素材 0.75%（p5 6.0，p50 14.9），仍在真实两个语料之间。
4. 放置语音在原对话里有他人同时说话的比例：AliMeeting 32.6%，AISHELL-5 21.5%。

耗时 9.5 分钟（设 `NUMPY_MADVISE_HUGEPAGE=0`）。按这个速度和窗口长度成正比估算，训练配比下每个在线窗口渲染约 0.36 秒。这是本机的估算，ACP 节点未实测。

## attempt 003 的结果（退出码 1，由下面第 2 条触发）

1. 泄漏通过：模板与素材都是训练会议，素材说话人不在 AliMeeting Eval/Test、AISHELL-5 Dev，也不在 zh245。AISHELL-4 只提供时间轴，其测试集说话人名单本地没有，只按会议核对。
2. 音频：无全零、无非有限值，峰值不超过 1，长度与窗口一致，无额外波形增强（`augment_calls` 为空）。2 个 300 s 窗口各有 1 帧 100 ms 低于 −120 dBFS（−173、−125 dBFS，前后约 −55 dBFS，非全零），真实 AISHELL-4 抽样 80 窗里也有 1 窗含静音帧，判定不影响训练。成因推测是噪声片段零段边界未被整帧剔除，未逐一核实。
3. 标注：格式、时间、说话人编号全部合法。
4. 标注处有语音：行能量距窗口底噪不足 3 dB 的比例（3 dB 是自定的线，只作对照）：

| | 不足 3 dB | p5 | p50 | 行数 |
| --- | --- | --- | --- | --- |
| 合成，AliMeeting 素材 | 1.41% | 5.0 | 14.2 | 24445 |
| 合成，AISHELL-5 素材 | 0.57% | 6.1 | 15.0 | 21151 |
| 真实，AliMeeting 远场（80 窗） | 0.36% | 10.9 | 19.5 | 3036 |
| 真实，AISHELL-4 远场（80 窗） | 2.87% | 3.5 | 9.4 | 1844 |

合成的声学范围由这两个真实语料合并而来，结果落在两者之间。

## 仍留着的风险

- 未标注的原对话他人：AliMeeting 素材 50.0%、AISHELL-5 素材 22.9% 的放置时长，在原对话里有他人同时说话。串音中位约 −22 dB（AISHELL-5 p90 −11.7 dB）。起点模型基本不转写它（0.1–3.5%），但它对训练的影响（是否教模型忽略较轻的语音）无法在训练前测出。
- 重配音的说话人归属差距、AISHELL-5 素材文字错误高：原因未定位，见 [../02-fidelity-check/README.md](../02-fidelity-check/README.md)。
