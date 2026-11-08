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
