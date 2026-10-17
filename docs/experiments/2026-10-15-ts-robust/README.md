# TS-ASR 真实场景鲁棒性续训（从 checkpoint-34479）

**状态：训练进行中（attempt 011），评测结果待补。** 目标是让目标说话人 ASR 在评测集 `ts_asr_benchmark` 暴露的场景里更稳：目标多段间歇说话、短促抢话、多个短干扰、相近声音、没有人声、真实噪声和房间。普通 ASR、警务、热词数据只作为「不劣于 34479」的门槛。

任务配比（已与业务确认）：目标说话人 50 / 普通 ASR 33 / 热词 17。

## 做了什么

| 部分 | 内容 |
| --- | --- |
| 在线混合 | 按评测条件混合：多段间歇（alternate）、短促抢话（burst）、多个短干扰（short）；干扰人是源文件的片段，取自记录和音频索引 |
| 素材池 | `ts_interference_pool`（干扰人）、`ts_nonspeech_pool`（无人声）、`ts_scene_pool`（多句场景与相近声音版）、`dns5_noise_train`（DNS5 噪声与房间响应），见 audio-data-contract `docs/datasets/ts-robust-pools.md` |
| 噪声和房间 | 加入道路交通、DNS5 噪声；噪声底改用 WHAM 加 DNS5 稳态片段；远场响应按直达声/混响比选取；加性噪声信噪比放宽到 0–20 dB |
| 记录来源 | 全部走已登记的 Lance 记录索引（`record_index: lance`），47 个来源与 jsonl 对账 0 差异 |
| 与评测隔离 | 评测说话人只来自官方 test 划分；训练跳过评测用的每包前 80 个 DNS5 噪声片段和哈希最小的 300 个响应。道路交通噪声与评测共用同一个池（噪声类型重叠，已在素材池文档说明） |

## 训练记录

配方在 `runs/ts-robust-acp-20261015/`（不入库）：`configs/train-data-v4.yaml`、`configs/train-1node-v4.yaml`（2 节点版 `train-2node-v4.yaml`）。

| 尝试 | 结果 |
| --- | --- |
| 005、007 | 被外部终止（SIGTERM / 挂起），原因未知；007 保存了 `checkpoint-3000` |
| 008（v3 场景配方） | 排队未调度，已停止，被 v4 取代 |
| 009、010 | 009（1 节点）和 010（2 节点）先后提交，010 一直卡在启动，改回 1 节点 |
| **011** | 1 节点，从 007 的 `checkpoint-3000` 续训，数据 `train-data-v4.yaml`（数据种子 1001），代码快照 `shared/code-v4`；W&B：`ASR-ts-robust-v4-hardneg-noise-lance-from007-ckpt3000-1node-011` |

011 的 loss 约 0.25–0.28（007 末期约 0.22），数据换成 v4 后略高，未见发散。

## 过程中查到的问题

- Lance 路径下，Lhotse 来源里「从文件开头起」的片段会被当成整段文件读，长度不符后报错；录音声明需比片段略长。已修（`supervision_cut`）并加回归测试。
- 第一版场景池有长达 75 s 的窗口，改为按模式限制场景总长（alternate 30 s、burst 22 s、short 14 s），新版本 `v2-20261016`。
- 评测 abs-hard 的相近度用 ECAPA（均值 0.53），训练用 ERes2NetV2（最近邻余弦中位数 zh 0.69、en 0.74），两个尺度不能直接比。

## 已知缺口

- 真实录音（`realreg`）只在评测里，没有对应的训练场景。
- 注册音频的噪声和混响只通过全局增强覆盖，没有逐条件的专门数据。
- 多句目标只在场景池里出现。
- 没有听审相近声音场景，只做了自动抽查。
- AISHELL-2、KeSpeech、LibriSpeech、Common Voice 的说话人质量列（`voiced_ratio` 等）仍为空。

`baselines/` 是 checkpoint-34479 和 Qwen3-ASR-1.7B 在普通 ASR、热词、TS 评测上的基线评测配置与交接文件（日志按仓库约定不入库）。
