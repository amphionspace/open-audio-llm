# 11 相似嗓音干扰人表

- 起因：任务 09 发现误报（目标不在场却出字）95% 是转写了干扰人，约一半来自与注册者嗓音相似的干扰人；在线混音的干扰人是随机抽的，很少相似。
- 做法：`scripts/build_lookalike_table.py` 用 WeSpeaker ECAPA-TDNN512-LM（基准集挑难负例用的同一模型）给干扰池每个片段和训练记录的每段注册音频算声纹，按注册段保留余弦最高的 32 个池片段，余弦 ≥ 0.75 视为同一人跳过。训练时 `OnlineMixConfig.lookalike_prob` 让每个干扰人以该概率从这 32 个里抽（`open_audio_llm.data.lookalike`）。
- 表：`/workspace/data/lookalike/v1-20261108/`（`SHA256SUMS`）。
  - `lookalike-zh-aishellmix.npz`：池 `ts_interference_pool@v3-20261018:zh` 319,999 片段；注册 `aishellmix@pack3-current-audio-record-20260910-punc-20261102:train_1spk` 206,851 段；第 1 近邻均值余弦 0.658，前 32 均值 0.606。
  - `lookalike-en-librimix.npz`：池 `…:en` 200,000 片段；注册 `librimix@960-current-audio-record-20260910-punc-20261102:train_1spk` 175,006 段；0.597 / 0.517。
- 第一次建表（`v1-span-bug/`）把池片段只按 cut id 记，但池里 23,298（中文）/ 14,826（英文）个 cut id 对应多个片段，查回的片段可能不是算声纹的那段；改为 `pool_clip_key`（cut id + 起止）后重建，读表时检查键唯一。

| 验证（`scripts/verify_lookalike.py`，`verify-{zh,en}.json`） | 中文 | 英文 |
|---|---|---|
| 抽 300 段注册重算，存储余弦与重算余弦最大差 | 0.0002 | 0.0002 |
| 第 1 近邻余弦 均值 / 中位数 | 0.657 / 0.658 | 0.595 / 0.585 |
| 随机池片段余弦 均值 / 中位数 | 0.121 / 0.114 | 0.060 / 0.055 |
| 第 1 近邻高于随机配对 90 分位的比例 | 100% | 100% |
| 训练加载器（lookalike_prob 1、目标缺席）抽到的干扰人在近邻列表内 | 100 / 100 | 100 / 100 |

## v1 的问题与同语料表（v2）

- GRPO v21 / v22（从 SFT 073 avg3 起，v15 配方）用 v1 表：v22 把相似嗓音负例提到约 5% 后，rollout 里新增 469 条目标缺席题只多出 11 组有梯度（模型几乎都答对），评测上误报 −75、漏识 +197，难子集 short-hard 54→57、abs-hard 21→18：只挪了门槛。
- 原因：注册来自 aishellmix / librispeech（干净朗读），干扰人来自另一个语料池（emilia 等真实录音），录音条件不同就足以判断"不是同一人"。同语料内随机配对的 ECAPA 余弦也明显更高（中文 0.26 对 0.12）。
- v2（`/workspace/data/lookalike/v2-samecorpus-20261108/`）：`build_lookalike_table.py --same-corpus --speaker-pattern '^([^_-]+)'`，池就是同一语料的注册片段，按 `source_record_id` 前缀的说话人编号排除本人；在数据配置里把语料的音频版本（`aishellmix@pack3-current-audio-record-20260910`、`librimix@960-current-audio-record-20260910`）登记为干扰源，权重 1e-6（只经表抽取，随机抽到可能是本人）。

| 验证（`verify-samecorpus-{zh,en}.json`） | 中文 | 英文 |
|---|---|---|
| 注册 / 说话人 | 206,851 / 1,990 | 175,006 / 2,338 |
| 近邻对里同一说话人 | 0 / 6,619,232 | 0 / 5,600,192 |
| 存储余弦与重算最大差 | 0.0002 | 0.0002 |
| 第 1 近邻余弦 均值 / 同语料随机 均值 | 0.70 / 0.26 | 0.59 / 0.09 |
| 加载器抽到的干扰人在近邻列表内 | 100 / 100 | 100 / 100 |
