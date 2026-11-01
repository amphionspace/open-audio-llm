# AmphionASR 2.0 草稿头投机解码（2026-10-31）

给 10.28 版本（054 avg3，Qwen3-ASR-1.7B）训练一层 EAGLE 草稿头，用 vLLM 0.18 投机解码加速。校验是贪心的，输出与原模型一致（差异只在批次不确定性量级），所以不需要重测准确率，只比速度。

## 结论

- **2500 步短训（4 张 A800、19 分钟）的草稿头已经可用**：每次校验平均接受 2.0 个 token（k=2）/ 2.2–2.4 个（k=3）；单路普通 ASR 请求延迟 −23%，TS 流式预览滞后 −21~−28%，final 请求 −19~−23%（16/48 路）。和编码器去同步修复叠加后相对收益不变。
- **对比 suffix**：普通 ASR（请求互不重复）suffix 几乎不被接受（接受长度 1.24），比无投机还慢；TS 流式回放里 suffix 因为 final 重放预览已转写的音频而在 final 上更快，预览滞后两者相当。草稿头不依赖重复内容，普通 ASR 与 TS 都有效，而且不需要新依赖（suffix 需要 arctic-inference）。
- **推荐 k=2**：k=3 接受长度高 0.2–0.3，但每步多算一个草稿，各项速度与 k=2 的差别都在噪声内；k=1 稍慢。普通 ASR 16 路吞吐 +20%、48 路 +7%（GPU 接近满载，额外计算抵掉大部分收益）。
- 正式训练（03，8 卡 20000 步，约 3 小时）配置已备好未启动；短训曲线在结束时仍在下降，正式草稿头的接受率应更高（需复测）。

## 路线与原因

选 **vLLM `method: eagle` + 标准 Llama 草稿检查点**（`EagleLlamaForCausalLM`，`vllm/model_executor/models/llama_eagle.py`），不选 `method: mtp`：

- eagle 草稿检查点 vLLM 0.18 原生能加载：`config.json` 写 `LlamaForCausalLM`、权重只有 `fc` 和 `layers.0.*`，embed_tokens / lm_head 自动共享目标模型的（日志 “Sharing target model embedding/lm_head weights”）。vLLM 侧零补丁、无需注册新模型类。
- mtp 路线要求草稿配置的 `model_type` 在 vLLM 的 `MTPModelTypes` 列表里、并由 `SpeculativeConfig.hf_config_override` 改写架构名，Qwen3-ASR 不在其中，需要注册配置类 + 模型类 + 绕开架构改写。
- 多模态守卫 `_raise_if_multimodal` 只在需要额外输入槽（draft_model 方法或 parallel drafting）时触发，eagle/mtp 都不触发；两条路线共同的唯一问题是 `EagleProposer.load_model` 对所有多模态目标读 `target_model.config.image_token_index`，Qwen3-ASR 的 thinker 配置没有它（实测 `AttributeError`）。解决：目标模型类里一行 `self.config.image_token_index = thinker_config.audio_token_id`（`src/open_audio_llm/tsasr/vllm_backend.py`，交付包 `runtime/vendor/qwen3_asr_vllm.py` 同一行），草稿头不读这个值。

训练与 vLLM 运行的结构逐项一致（`src/open_audio_llm/integrations/ms_swift/draft_head.py`，单测对照 transformers 的 Llama 层与按 vLLM 语义逐位置模拟的多步草稿）：

- 隐状态契约：vLLM 传给草稿的是目标模型 **final norm 之后** 的最后一层输出（与送进 lm_head 的同一个张量），位置不变；草稿第 i 行 = (目标在 i 的隐状态, 第 i+1 个 token 的 embedding)，RoPE 位置 i（目标的 MRoPE 三维位置相同，草稿用第 0 维），预测第 i+2 个 token。
- 音频占位：`EagleLlamaForCausalLM.embed_input_ids` 不接受多模态参数，vLLM 回退为文本草稿（日志 “falling back to text-only mode”），音频位置用 `<|audio_pad|>` 的 embedding；训练里同样用 `embed_tokens(input_ids)`。
- 结构：`fc(cat(embed, hidden))` → 去掉 input_layernorm 的 Llama 层（无 q/k norm，rope_theta 1e6，GQA 16/8，head_dim 128）→ `h + residual` → 共享 lm_head（与 embed 绑定）；没有 final norm。
- 多步：第 k 步用上一步的输出特征和上一步的草稿 token，位置 +1，注意力看验证过的前缀（第一遍的 KV）和本次提案前几步的 KV；训练按这个掩码把 3 步并行展开（EAGLE-3 的 training-time test）。
- 目标：冻结模型（no_grad、eval）对下下个 token 的分布做 KL，只在 assistant 回复内（输入 token 与被预测 token 都是生成的）计算；第 k 步权重 0.8^(k−1)。只有草稿头约 6000 万参数可训练。

## 任务

| 任务 | 内容 | 状态 |
|---|---|---|
| [01-validate-draft](tasks/01-validate-draft/README.md) | 草稿头短训（ACP 开发节点 GPU 4–7，2500 步，不接 W&B） | 完成 |
| [02-serve-bench](tasks/02-serve-bench/README.md) | vLLM 无损、接受率、并发 1/16/48 速度，对照无投机与 suffix，另测叠加去同步修复 | 完成，W&B 离线待补录 |
| [03-train-draft](tasks/03-train-draft/README.md) | 正式草稿头训练（`configs/train-draft.yaml`） | 未启动 |

## 风险

- 骨干换成 GRPO 等新权重后必须用新权重重训草稿头（只改 `--model`）；旧草稿头接到新骨干上仍无损，但接受率会降，未测。
- 速度只在独占的 A800 上测过；线上是 RTX PRO 4500 Blackwell 4 路分时共享，草稿头每步多一次小前向，分时 GPU 上的净收益需在 Focus test 复测。
- 训练数据是标注文本（teacher forcing），不是骨干自己的贪心输出；骨干输出偏离标注的地方草稿头学得少，接受率可能低估潜力。
- vLLM 的普通 ASR 内置类（`vllm/model_executor/models/qwen3_asr.py`，仓库 serve 在不插 SEP 时用它）没有这一行，要在那条路径上开投机需同样处理；交付包两种模式都走自带的类，不受影响。
