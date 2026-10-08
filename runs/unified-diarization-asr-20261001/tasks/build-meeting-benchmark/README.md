# 180 秒会议评测集

按部署方式（会议音频按约 180 秒切段送入模型）评测会议转写，覆盖中文、英文和 8 人以上场景。与训练集、选模集均不重叠。

## 构成（执行 003，共 424 段、21.2 小时）

| 部分 | 来源 | 段数 | 时长 | 每段说话人数 |
| --- | --- | ---: | ---: | --- |
| 中文真实会议 | AISHELL-4 test（20 场） | 120 | 6.0 h | 1–7（6 人 17 段、7 人 4 段） |
| 中文真实会议 | AliMeeting Test（20 场） | 120 | 6.1 h | 2–4 |
| 英文真实会议 | AMI SDM test（16 场） | 94 | 4.7 h | 2–4 |
| 合成会议 | 中文 / 英文 / 中英混合 | 各 30 | 各约 1.5 h | 6、8、10 各 10 段 |

- 真实会议：每场均匀取 6 个窗口；在 150–210 秒内的说话空隙处切，不切断句子；无空隙时把跨界句子完整并入（共 65 个窗口，最长 244 秒）。远场通道取均值。
- 合成会议：AISHELL test / LibriSpeech test-clean 的说话人（训练未见），轮流发言、约 20% 句子与前一句重叠；每人一条真实房间冲激响应（SLR28），加 MUSAN 噪声（SNR 10–20 dB）。
- 推理计划：与固定集相同的 vLLM 协议，编码器整段注意力上限调到 26000（260 秒）。

## 局限

- 合成档的噪声与训练加噪同为 MUSAN，抗噪结论偏乐观；此档主要测说话人数容量。
- 真实会议切点对齐说话空隙，部署时若在句中切段会更难。
- 没有 8 人以上的真实会议；CHiME-6 测试集（2 场多阵列晚宴）未纳入。

配置：[configs/build-meeting-benchmark.yaml](../../configs/build-meeting-benchmark.yaml)；脚本：[scripts/build_meeting_benchmark.py](../../scripts/build_meeting_benchmark.py)。音频与清单在 `attempts/003/artifacts/`（不入 Git）。

## 团队共享登记（2026-10-08）

- 数据集：audio-data-contract `meeting_180s_benchmark@v1-20261008`（PR #19），6 个划分对应上表；portable AudioRecord（`speaker_attributed_asr`）+ 音频索引，根目录别名 `meeting_180s_benchmark` 指向数据集目录（本机 `/workspace/data/datasets/meeting_180s_benchmark`）。导出见 [export-meeting-benchmark-dataset](../../configs/export-meeting-benchmark-dataset.yaml)（执行 004）。
- 评测集：AmphionEval `open-audio-llm/meeting-180s@20261008`（MR !17），已发布到团队 COS；登记内容见 [eval-set-meeting-180s.yaml](../../configs/eval-set-meeting-180s.yaml)。
- 备份：`whai:open-audio-llm/datasets/meeting_180s_benchmark/`（431 个文件，与本机核对一致）。其他机器下载后在 `AUDIO_DATA_ROOTS_FILE` 中把别名指向下载位置，训练后评测写 `eval_set: open-audio-llm/meeting-180s@20261008`。
- `ae run --manifest` 需要的 Lhotse cut 清单含本机绝对路径，不进 catalog；本机副本在导出执行的 `artifacts/cuts/`。
