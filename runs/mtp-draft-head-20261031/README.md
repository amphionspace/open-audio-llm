# AmphionASR 2.0 草稿头投机解码（2026-10-31）

给 Qwen3-ASR-1.7B 骨干（实验用 10.28 版本权重）训练一层草稿头，用 vLLM 0.18 投机解码加速。校验是贪心的，骨干全程冻结，所以不重测准确率，只比速度。最终草稿头会在 GRPO 之后的权重上重训。

## 结论

- **最终设计：EAGLE-3，融合骨干第 2 / 14 / 25 层，用标注文本训练。** 留出请求上，96 万条样本（10000 步）时每次校验接受 2.72 / 2.66 个 token（k=2，普通 ASR / TS）、3.43 / 3.31 个（k=3），k1–k3 都高于 EAGLE-1（k=3 为 3.36 / 3.14）。预算小时结论相反：24 万条时 EAGLE-1 领先，48 万条时已接近。
- **自蒸馏不采用**：用模型贪心输出微调和用标注文本微调，接受长度差别最大 0.011，方向不一致，成本约 17 倍。只比了 4.8 万条的微调，不是从头训练。
- **无损**：单路时所有投机变体的输出彼此逐字节相同（覆盖两种结构、k1–k3、2500 至 10000 步），与无投机都只差同样的 3 / 306 条 ASR、2 / 300 条 TS。差异来自校验前向的计算路径，与草稿头无关，数量与无投机自己换并发同级。
- **速度**：已测的是 2500 步草稿头（独占 A800）：普通 ASR 单路 p50 78→61 ms，16 路吞吐 +19%，48 路 +7%；TS 流式预览滞后 −23~−26%（叠加去同步修复时 −20~−28%）。5000 步 k=3 单路：ASR p50 56–59 ms，TS 58–60 ms。10000 步草稿头（015–018，GPU 独占，CPU 与存储与 GRPO 共用）：普通 ASR 单路均值 100→67 ms（EAGLE-3 k=3），16 路吞吐 1122→1471 token/s，48 路 1968→2148；TS 单路 83→64 ms；TS 流式 48 路预览滞后 p50 / p95 172 / 496 → 119 / 378 ms。选定 **EAGLE-3 k=3**。流式 TS 上训练的草稿头与不用训练的 suffix 相当。
- **对比 suffix**：普通 ASR 上 suffix 接受长度只有 1.24，比无投机还慢；TS 流式里 final 重放预览已转写的音频，suffix 在 final 上比 2500 步草稿头更快（10000 步草稿头尚未对比）。
- 正式训练配置已备好未启动（03），提交前只需换 `--model`。

## 路线与原因

**vLLM `method: eagle3` + 标准 `LlamaForCausalLMEagle3` 草稿检查点**（`vllm/model_executor/models/llama_eagle3.py`）。vLLM 不打补丁、不注册新模型类，只在我们的目标模型类里改两处（`src/open_audio_llm/tsasr/vllm_backend.py`）：

- `self.config.image_token_index = audio_token_id`：`EagleProposer.load_model` 对所有多模态目标都读这个属性，Qwen3-ASR 配置没有它会 `AttributeError`，草稿头不用它。
- 基类加 `SupportsEagle3`：`Qwen3Model` 本来就带 `EagleModelMixin`，默认的 `set_aux_hidden_state_layers` 走 `get_language_model().model`，`forward` 把 `(hidden, aux)` 原样返回。
- 多模态守卫 `_raise_if_multimodal` 只在 draft_model 方法或 parallel drafting 时触发，eagle / eagle3 不触发。

训练与 vLLM 运行的结构逐项一致（`src/open_audio_llm/integrations/ms_swift/draft_head.py`，单测按 vLLM 语义逐位置模拟多步草稿并对照）：

- 隐状态契约：vLLM 取 `hidden + residual`，id `j` 是第 `j - 1` 层之后、即进入第 `j` 层的残差流，在 final norm 之前；默认 `(2, L // 2, L - 3)`，28 层时为 2 / 14 / 25，写进检查点的 `eagle_aux_hidden_state_layer_ids`。训练用 hook 取同一张量（单测对照 transformers 的 `hidden_states`），位置不变。
- 第一遍第 i 行 = `fc(cat(aux_2, aux_14, aux_25))` 与第 i+1 个 token 的 embedding，RoPE 位置 i（MRoPE 三维位置相同，取第 0 维），预测第 i+2 个 token。
- 音频占位：`Eagle3LlamaForCausalLM.embed_input_ids` 接受多模态参数但忽略它，音频位置用 `<|audio_pad|>` 的 embedding，音频信息来自隐状态；训练同样用 `embed_tokens(input_ids)`。
- 结构：`cat(input_layernorm(embed), hidden_norm(h))` → 注意力（残差为 `h`，GQA 16/8，head_dim 128，rope_theta 1e6，无 q/k norm）→ MLP → final `norm` → 共享 lm_head。后续步喂 final norm 之前的输出，不再过 `fc`。
- 多步：第 k 步位置 +1，注意力看验证过的前缀（第一遍的 KV）和本次提案前几步的 KV；训练按这个掩码把 3 步并行展开（EAGLE-3 的 training-time test）。
- 目标：冻结模型（no_grad、eval）对下下个 token 的分布做 KL，只在 assistant 回复内计算；第 k 步权重 0.8^(k−1)。只有草稿头 7130 万参数可训练。

## 任务

| 任务 | 内容 | 状态 |
|---|---|---|
| [01-validate-draft](tasks/01-validate-draft/README.md) | 草稿头短训（ACP 开发节点 GPU 4–7，2500 / 5000 / 10000 步，自蒸馏 A/B） | 完成 |
| [02-serve-bench](tasks/02-serve-bench/README.md) | vLLM 无损、接受率、并发 1/16/48 速度，对照无投机、suffix、EAGLE-1，另测叠加去同步修复 | 完成；10000 步测速选定 EAGLE-3 k=3 |
| [03-train-draft](tasks/03-train-draft/README.md) | 正式草稿头训练（`configs/train-draft.yaml`） | 未启动 |

## 风险

- 384 万条的正式预算没有实测接受率；k=3 由 10000 步草稿头的测速选定。
- 骨干换成 GRPO 权重后必须重训（只改 `--model`）；旧草稿头接到新骨干上仍无损，但接受率会降，未测。
- 速度只在独占 A800 上测过；线上是分时共享的卡，草稿头每步多一次小前向，净收益要在目标环境复测。
- 接受率样本是 306 条普通 ASR + 300 条 TS，取自评测集，不是全量；数字用于比较设计，不是全量评测结果。
- 自蒸馏结论只覆盖微调阶段。
- vLLM 内置 Qwen3-ASR 类（仓库 serve 在不插 SEP 时用）没有这两处改动，那条路径开投机要同样处理；交付包两种模式都走自带的类。
- 单路无损差异里有一条 TS 缺席样本从不出字变成出字，属于近似并列翻转，无投机换并发时同样出现。
