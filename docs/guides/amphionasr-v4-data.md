# AmphionASR v4 数据配方

从官方 Qwen3-ASR-1.7B 做一轮全参 SFT。数据配置是
[amphionasr_v4.yaml](../../examples/configs/data/amphionasr_v4.yaml)，本机入口是
[amphionasr-v4.yaml](../../examples/configs/train/amphionasr-v4.yaml)，ACP 入口是
[amphionasr-v4-acp.yaml](../../examples/configs/train/amphionasr-v4-acp.yaml)。

配方不设 `replay`。各来源按登记条数交织，`reps` 把同一批记录在一轮里重复相应次数。
时长来自 Catalog 划分上的 `statistics.duration_hours`。目标说话人只累计混合音，不含注册音。
`fleurs_en@legacy` 的登记没有写入时长，下表用其 supervision 清单的时长合计。

一轮合计 **107,576.01 小时**、**54,230,807 条**（条数已乘 `reps`）。

| 任务 | 小时 | 占一轮 |
|---|---:|---:|
| 目标说话人 `ts_asr` | 64,731.01 | 60.2% |
| 普通 ASR | 33,254.73 | 30.9% |
| 热词 ASR | 9,590.27 | 8.9% |

Emilia2 的 short、dialog、mix 三个训练集合计 53,956.49 小时，占一轮的 50.2%。

## 登记位置

数据 YAML 的 `catalog` 指向 `data/amphionasr/catalog`，`roots` 指向同目录的 `roots.json`。
登记格式是 `dataset-catalog/2.0`。下表的清单路径相对于对应 root。

| root | 本机目录 |
|---|---|
| `aidc_data`、`legacy_asr`、`multilingual`、`shared_storage` | `/workspace/data` |
| `amphionasr` | `/workspace/workspace/lx/data/amphionasr` |

AudioRecord 来源带 `audio-records` 和 `audio-index`。Lhotse 来源带 recordings 与 supervisions，任务在数据 YAML 里写成 `task: asr`。目标说话人和热词的任务写在记录上。

## 目标说话人

每条有两段音频：先是注册音，再是混合音，模型只转写注册说话人所说的内容。负样本的目标为空。
AISHELLMix、LibriMix 的四个 train 划分各进一整遍，其中包含 `train_1spk`。

| 数据集 @ 版本 | 划分 | 条数 | 混合音小时 | 重复 | 登记文件 |
|---|---|---:|---:|---:|---|
| `emilia2_mix@mix-20261013` | `train_2spk` | 2,208,856 | 8,566.00 | 1 | `catalog/emilia2_mix.yaml` |
| 同上 | `train_3spk` | 2,204,373 | 8,566.07 | 1 | 同上 |
| 同上 | `train_neg` | 4,413,261 | 17,132.14 | 1 | 同上 |
| `emilia2_dialog@dialog-turn-taking-20261013-enroll` | `train_2spk` | 1,129,151 | 6,719.79 | 1 | `catalog/emilia2_dialog.yaml` |
| 同上 | `train_3spk` | 439,583 | 2,691.16 | 1 | 同上 |
| 同上 | `train_neg` | 890,340 | 5,178.40 | 1 | 同上 |
| `aishellmix@pack3-current-audio-record-20260910` | `train_1spk` | 326,176 | 1,011.65 | 1 | `catalog/aishellmix.yaml` |
| 同上 | `train_2spk` | 652,200 | 2,343.73 | 1 | 同上 |
| 同上 | `train_3spk` | 978,300 | 3,769.42 | 1 | 同上 |
| 同上 | `train_neg` | 97,830 | 364.35 | 1 | 同上 |
| `librimix@960-current-audio-record-20260910` | `train_1spk` | 273,541 | 955.78 | 1 | `catalog/librimix.yaml` |
| 同上 | `train_2spk` | 543,842 | 2,177.96 | 1 | 同上 |
| 同上 | `train_3spk` | 812,217 | 3,396.18 | 1 | 同上 |
| 同上 | `train_neg` | 82,020 | 335.58 | 1 | 同上 |
| `alimeeting_sdm@turn-taking-20261014` | `train_2spk` | 10,020 | 72.29 | 1 | `catalog/alimeeting_sdm.yaml` |
| 同上 | `train_3spk` | 47,348 | 191.84 | 1 | 同上 |
| 同上 | `train_neg` | 26,581 | 119.19 | 1 | 同上 |
| `aishell4@turn-taking-20261014` | `train_2spk` | 10,506 | 73.88 | 1 | `catalog/aishell4.yaml` |
| 同上 | `train_3spk` | 21,925 | 135.98 | 1 | 同上 |
| 同上 | `train_neg` | 14,128 | 91.37 | 1 | 同上 |
| `ami_sdm@turn-taking-20261014` | `train_2spk` | 5,081 | 30.20 | 1 | `catalog/ami_sdm.yaml` |
| 同上 | `train_3spk` | 23,853 | 125.33 | 1 | 同上 |
| 同上 | `train_neg` | 13,117 | 68.33 | 1 | 同上 |
| `magicdata_ramc@turn-taking-20261014` | `train_2spk` | 29,508 | 221.88 | 1 | `catalog/magicdata_ramc.yaml` |
| 同上 | `train_neg` | 15,372 | 115.21 | 1 | 同上 |
| `chime6@turn-taking-20261014` | `train_2spk` | 1,181 | 3.61 | 2 | `catalog/chime6.yaml` |
| 同上 | `train_3spk` | 25,258 | 96.34 | 2 | 同上 |
| 同上 | `train_neg` | 10,959 | 38.70 | 2 | 同上 |

登记说明：

- `emilia2_mix`：Emilia2 long 合成的 2/3 人混合。清单在 `aidc_data:datasets/emilia2-mix/versions/mix-20261013/records/`。注册音是同一录音上没用过的 1～30 秒片段。
- `emilia2_dialog`：对话 turn-taking，音频用 m4a-128k，只保留纯中文或纯英文。混合窗 1～30 秒。注册音是同一说话人的连续整句，1～30 秒。清单在 `amphionasr:datasets/emilia2_dialog/versions/dialog-turn-taking-20261013-enroll/records/`。
- `aishellmix`、`librimix`：已有 AudioRecord。独立的注册音和混合音，负样本转写为空。清单分别在 `aidc_data:datasets/aishellmix/.../pack3-current-audio-record-20260910/` 和 `aidc_data:datasets/librimix/.../960-current-audio-record-20260910/`。
- 五套会议话轮：整句拼成 2/3 人混合窗，时长 0.5～30 秒；注册音是窗外另一句，1～30 秒。RAMC 没有 3 人划分。清单在 `amphionasr:datasets/<数据集>/versions/turn-taking-20261014/records/`。来源分别是 `alimeeting_sdm@icefall-20260908` 加已登记近场、AISHELL-4 的 legacy `train_L`/`test` 与 icefall `train_M`/`train_S`、`ami_sdm@icefall-20260908`、`magicdata_ramc@legacy-20260804`、`chime6@speaker-records-v1-20260912`。

## 普通 ASR

| 数据集 @ 版本 | 划分 | 条数 | 小时 | 重复 | 登记文件 |
|---|---|---:|---:|---:|---|
| `gigaspeech@legacy-20260804` | `train` | 8,282,988 | 9,997.82 | 1 | `catalog/replay/gigaspeech.yaml` |
| `wenetspeech_youtube@l-20261014` | `train` | 8,289,968 | 6,113.14 | 1 | `catalog/replay/wenetspeech_youtube.yaml` |
| `wenetspeech_podcast@l-20261014` | `train` | 6,335,277 | 3,892.28 | 1 | `catalog/replay/wenetspeech_podcast.yaml` |
| `emilia2_short@short-20261014` | `train_zh` | 1,913,616 | 2,362.76 | 1 | `catalog/emilia2_short.yaml` |
| 同上 | `train_en` | 2,081,298 | 2,740.17 | 1 | 同上 |
| `common_voice_en@legacy-20260804` | `train` | 1,143,276 | 1,808.18 | 1 | `catalog/replay/common_voice_en.yaml` |
| `kespeech@legacy-20260804` | `train` | 884,021 | 1,396.94 | 1 | `catalog/replay/kespeech.yaml` |
| `police_asr_unified@v1-20261009` | `train` | 365,341 | 657.92 | 2 | `catalog/police_asr_unified.yaml` |
| `vitw@2m-20261013` | `train` | 556,414 | 1,085.41 | 1 | `catalog/vitw.yaml` |
| `aishell2@legacy-20260804` | `train` | 1,009,223 | 1,000.76 | 1 | `catalog/replay/aishell2.yaml` |
| `librispeech@legacy-20260804` | `train` | 281,241 | 961.05 | 1 | `catalog/replay/librispeech.yaml` |
| `alimeeting_sdm@asr-20261014` | `train` | 152,368 | 137.27 | 1 | `catalog/alimeeting_sdm.yaml` |
| `aishell4@asr-20261014` | `train` | 96,999 | 105.36 | 1 | `catalog/aishell4.yaml` |
| `ami_sdm@asr-20261014` | `train` | 85,414 | 76.90 | 1 | `catalog/ami_sdm.yaml` |
| `magicdata_ramc@asr-20261014` | `train` | 154,195 | 123.99 | 1 | `catalog/magicdata_ramc.yaml` |
| `common_voice_zh@legacy-20260804` | `train` | 29,560 | 42.59 | 2 | `catalog/replay/common_voice_zh.yaml` |
| `fleurs_zh@legacy-20260804` | `train` | 3,246 | 9.69 | 3 | `catalog/replay/fleurs_zh.yaml` |
| `fleurs_en@legacy` | `train` | 2,602 | 7.53 | 3 | `catalog/replay/fleurs_en.yaml` |

登记说明：

- `catalog/replay/` 里的 GigaSpeech、LibriSpeech、Common Voice、KeSpeech、AISHELL-2、FLEURS 是已有 legacy 登记。清单仍在 `legacy_asr:LHOTSE/...` 或 `multilingual:.../fleurs_*/data/manifests/`，由 `roots.json` 指到 `/workspace/data`。
- WenetSpeech 两份是 L 全集按录音路径拆成 YouTube 和 podcast，不是 3000 小时抽样。M、S 已包含在 L 里，没有再加。清单在 `amphionasr:datasets/wenetspeech_{youtube,podcast}/versions/l-20261014/`。
- `emilia2_short` 使用已抽出的 m4a-128k short。只保留语言标签为纯中文或纯英文、时长 0.5～30 秒的片段。不使用 m4a-64k 和 asmr。清单在 `amphionasr:datasets/emilia2_short/versions/short-20261014/records/`。
- `police_asr_unified` 合并已登记的警务合成批次，不复制音频。清单在 `aidc_data:datasets/police_asr_unified/versions/v1-20261009/manifests/`。不含 `police_incremental_20260916`。
- `vitw` 是 2M 配方全量，音频留在原目录。清单在 `amphionasr:datasets/vitw/versions/2m-20261013/`。
- 四套会议普通转写是每条 supervision 一条，时长 0.5～30 秒，文本非空，多通道只取第 0 路。CHiME-6 没有普通转写划分。清单在 `amphionasr:datasets/<数据集>/versions/asr-20261014/records/`。

## 热词

每次都给出热词列表：`hotword_prompt_prob: 1.0`，`hotword_miss_prob: 0.0`，`max_hotwords: 30`。列表仍会补入干扰词。`train_hotwords` 只保留系统提示里带 `Hotwords:` 的行。

| 数据集 @ 版本 | 划分 | 条数 | 小时 | 重复 | 登记文件 |
|---|---|---:|---:|---:|---|
| `hw_sft@hw-20261014` | `train_hotwords` | 834,336 | 1,171.44 | 8 | `catalog/hw_sft.yaml` |
| `hard_hw_sft@hard-20261014` | `train` | 69,275 | 109.36 | 2 | `catalog/hard_hw_sft.yaml` |

音频指向已登记的源文件，没有另存一份音频。清单分别在 `amphionasr:datasets/hw_sft/versions/hw-20261014/records/train_hotwords.jsonl.gz` 和 `amphionasr:datasets/hard_hw_sft/versions/hard-20261014/records/train.jsonl.gz`。来源 jsonl 在 `data/ckpt34479_recipe/AmphionASR-1.7B-v3/data/sharegpt/hw_sft/`。`hw_sft` 里系统提示为空的 208,981 条不在 `train_hotwords` 中。

## 验证

验证只看 loss。训练器按各来源条数比例抽 **54,000** 条，种子是数据 YAML 的 `seed: 42`，每轮同一批。抽之前丢掉长于 `validation_max_duration: 120` 秒的片段。验证关闭下面的波形增强和随机热词遗漏。

登记的全量划分如下。小时仍是混合音或 supervision 时长，抽到的 54,000 条是其中的子集。

| 数据集 @ 版本 | 划分 | 条数 | 小时 | 登记文件 |
|---|---|---:|---:|---|
| `emilia2_dialog@dialog-turn-taking-20261013-enroll` | `eval_2spk` / `eval_3spk` / `eval_neg` | 22,323 / 9,368 / 18,069 | 132.52 / 56.65 / 104.48 | `catalog/emilia2_dialog.yaml` |
| `emilia2_mix@mix-20261013` | `eval_2spk` / `eval_3spk` / `eval_neg` | 49,139 / 48,206 / 97,345 | 185.95 / 186.42 / 372.36 | `catalog/emilia2_mix.yaml` |
| `alimeeting_sdm@turn-taking-20261014` | `eval_2spk` / `eval_3spk` / `eval_neg` | 376 / 1,361 / 822 | 2.79 / 6.44 / 4.23 | `catalog/alimeeting_sdm.yaml` |
| `ami_sdm@turn-taking-20261014` | `eval_2spk` / `eval_3spk` / `eval_neg` | 236 / 1,437 / 746 | 1.42 / 7.29 / 3.78 | `catalog/ami_sdm.yaml` |
| `magicdata_ramc@turn-taking-20261014` | `eval_2spk` / `eval_neg` | 1,814 / 965 | 13.45 / 7.12 | `catalog/magicdata_ramc.yaml` |
| `chime6@turn-taking-20261014` | `eval_2spk` / `eval_3spk` / `eval_neg` | 74 / 3,040 / 1,214 | 0.25 / 12.53 / 4.79 | `catalog/chime6.yaml` |
| `alimeeting_sdm@asr-20261014` | `eval` | 5,193 | 4.77 | `catalog/alimeeting_sdm.yaml` |
| `ami_sdm@asr-20261014` | `eval` | 4,955 | 4.44 | `catalog/ami_sdm.yaml` |
| `magicdata_ramc@asr-20261014` | `eval` | 9,761 | 8.12 | `catalog/magicdata_ramc.yaml` |
| `police_asr_unified@v1-20261009` | `dev` | 36,086 | 65.30 | `catalog/police_asr_unified.yaml` |
| `gigaspeech@legacy-20260804` | `dev` | 6,750 | 12.49 | `catalog/replay/gigaspeech.yaml` |
| `common_voice_en@legacy-20260804` | `dev` | 16,401 | 27.34 | `catalog/replay/common_voice_en.yaml` |
| `common_voice_zh@legacy-20260804` | `dev` | 10,647 | 16.01 | `catalog/replay/common_voice_zh.yaml` |
| `kespeech@legacy-20260804` | `dev` | 4,407 | 6.83 | `catalog/replay/kespeech.yaml` |
| `librispeech@legacy-20260804` | `dev` | 5,567 | 10.51 | `catalog/replay/librispeech.yaml` |
| `fleurs_zh@legacy-20260804` | `dev` | 409 | 1.28 | `catalog/replay/fleurs_zh.yaml` |
| `fleurs_en@legacy` | `validation` | 394 | — | `catalog/replay/fleurs_en.yaml` |

数据 YAML 末尾的 `evaluation` 列出各来源已登记的 test 划分。训练器不读这一段。

## 增强

只在训练时、按每段音频槽位独立抽样。目标说话人的注册音和混合音各自抽一次。顺序是变速、一次混响、加噪。信噪比用混响之后的信号功率。验证集 `training=False`，不加这些变换。`speed_prob` 和 `spec_aug_prob` 未设置，因此为 0。

| 项 | 取值 |
|---|---|
| 普通混响 `rir_prob` | 0.25 |
| 远场混响 `far_rir_prob` | 0.25 |
| 不加混响 | 0.50 |
| 加噪 `noise_prob` | 0.50 |
| 信噪比 `noise_snr_db` | 均匀采样 5–20 dB |

普通混响和远场混响互斥，一条音频最多卷积一次。远场分支从下面三份已登记 RIR 里抽子集，三份来源等概率，来源内部再均匀抽一条。

| 数据集 @ 版本 | 划分 | 条数 | 权重 | 登记文件 | 清单 |
|---|---|---:|---:|---|---|
| `real_rir_slr28@prepared-20260908` | `all` | 178 | 等概率 | `catalog/rir/real_rir_slr28.yaml` | `aidc_data:datasets/real_rir_slr28/versions/prepared-20260908/recordings/real_rir_slr28_recordings_all.jsonl.gz` |
| `sim_rir_slr26@legacy-20260804` | `all` | 56,168 | 等概率 | `catalog/rir/sim_rir_slr26.yaml` | `aidc_data:datasets/sim_rir_slr26/versions/legacy-20260804/recordings/sim_rir_slr26_recordings_all.jsonl.gz` |
| `sim_rir_slr28@legacy-20260804` | `all` | 56,169 | 等概率 | `catalog/rir/sim_rir_slr28.yaml` | `aidc_data:datasets/sim_rir_slr28/versions/legacy-20260804/recordings/sim_rir_slr28_recordings_all.jsonl.gz` |

远场子集：真实 RIR 保留文件名含 `far` 且不含 `smallroom` 的记录。模拟 RIR 先取接近 7 m × 5 m × 3 m、Sabine T60 在 0.4–0.6 秒的小房间，再在每个房间里保留直达混响比最接近 0 dB 的那一部分响应。

加噪时先按权重抽来源，再在来源内部均匀抽一条。噪声本身不再卷积。三份远场人声在制作时已经卷过房间响应；WHAM 和 MUSAN 保持干声。

| 数据集 @ 版本 | 划分 | 条数 | 权重 | 登记文件 | 清单 |
|---|---|---:|---:|---|---|
| `wham@tr-20261014` | `train` | 20,000 | 30 | `catalog/noise/wham.yaml` | `amphionasr:datasets/wham/versions/tr-20261014/recordings/wham_tr_recordings.jsonl.gz` |
| `musan@legacy-20260804` | `noise` | 930 | 17.5 | `catalog/noise/musan.yaml` | `aidc_data:datasets/musan/versions/legacy-20260804/noise_recordings/musan_recordings_noise.jsonl.gz` |
| `farfield_speech_noise_1spk@20261014` | `train` | 10,000 | 17.5 | `catalog/noise/farfield_speech_noise_1spk.yaml` | `amphionasr:datasets/farfield_speech_noise_1spk/versions/20261014/records/` |
| `farfield_speech_noise_2spk@20261014` | `train` | 10,000 | 17.5 | `catalog/noise/farfield_speech_noise_2spk.yaml` | `amphionasr:datasets/farfield_speech_noise_2spk/versions/20261014/records/` |
| `farfield_speech_noise_3spk@20261014` | `train` | 10,000 | 17.5 | `catalog/noise/farfield_speech_noise_3spk.yaml` | `amphionasr:datasets/farfield_speech_noise_3spk/versions/20261014/records/` |

WHAM 的 wav 指向 `/workspace/data/datasets/noise_open/wham_noise/extracted/wham_noise/tr/`。MUSAN 只用 noise，不含 music 和 speech。远场人声噪声由 Emilia 中文或英文干声卷上远场房间响应得到；2 人、3 人是不同说话人错开开口后叠加。1/2/3 人的文件时长合计分别是 25.50、30.35、32.73 小时。

## 组批

`batching.max_duration: 480` 与 `max_samples: 12` 同时生效。30 个时长桶，每次从 2000 条里分桶。再加一条会超过 480 秒时，不满 12 条也封批。`drop_last: false`。8 卡、每卡 batch 12、梯度累积 1 时，一轮优化器步数是 581,081。
