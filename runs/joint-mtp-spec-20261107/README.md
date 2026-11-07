# 联合训练 MTP 的投机解码收益（2026-11-07）

v16 avg3（新提示词，SFT 073 起，GRPO 两轮）带一个随训练一起学的 MTP 模块（`checkpoint/mtp/`）。这里用 vLLM 0.18 把它当投机解码草稿，测接受率和速度。校验是贪心的，主模型输出不应改变。

## 结论

- **MTP 能直接当草稿用，接受率高于单独训练的 EAGLE-3 草稿头**：k3 每次校验接受 3.62 / 3.51 个 token（普通 ASR / TS），逐位置 0.93 / 0.87 / 0.81；EAGLE-3 草稿头（10000 步）为 3.43 / 3.31。
- **速度**：单路普通 ASR 均值 103→63 ms（k3），TS 90→62 ms；普通 ASR 吞吐 16 路 +35%、48 路 +15%，TS 48 路 +13%。综合推荐 k2。
- **无损**：与无投机的差异量级同无投机自己换并发，与之前 EAGLE 实验一致。

## 路线

- vLLM 自带的 MTP 路径只认 `MTPModelTypes` 里的模型和 Qwen3-Next 的门控 / MoE 层；我们的模块是一层普通 Qwen3 层，所以走 EAGLE 提议器（`method: "eagle"`），草稿类 `EagleQwen3ASRMTP`（`src/open_audio_llm/tsasr/vllm_mtp.py`）。
- EAGLE 提议器给草稿的输入与训练一致：目标模型 final norm 之后的隐状态、下一个 token 的纯文本 embedding、一维位置；多步时把草稿自己的输出隐状态喂回同一层。embed_tokens 与 lm_head 共享目标模型的。
- `mtp/` 目录加上 `write_vllm_config` 写入的 Qwen3 字段就是草稿目录；新训练的 checkpoint 由 `save_mtp` 直接写好。
- 服务包 `shared/pack-mtp` = 11.06 交付包 runtime + EAGLE 所需的两处目标类改动（同 mtp-draft-head-20261031 的 pack-eagle）+ `vendor/qwen3_asr_mtp.py`（`vllm_mtp.py` 原样复制）并注册。

## 任务

| 任务 | 内容 | 状态 |
|---|---|---|
| [01-spec-bench](tasks/01-spec-bench/README.md) | 无投机 / MTP k1 / k2 / k3，普通 ASR 306 条 + TS 300 条，并发 1/16/48；004 为草稿迁移到合并模型 | 完成（003、004） |
| [02-stream-bench](tasks/02-stream-bench/README.md) | TS 流式回放，无投机 / k2 / k3 | 完成：预览滞后 p50 −30%~−38% |
