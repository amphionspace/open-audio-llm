# 03-states：决策位置的白盒状态提取

## 输入

- Checkpoint：`/workspace/data/ae-runs/amphionasr2-grpo-v16avg3-ckpt400/checkpoint`（新 TS prompt；只用 `model.safetensors`，不用 `mtp/`）。
- AmphionEval run（vLLM 0.18.0，同一 checkpoint）：`/workspace/data/ae-runs/amphionasr2-grpo-v16avg3-ckpt400/runs/run-20261107T022131Z-c6ec67e6/`，即 ts-asr-benchmark v12，26,750 条（13,797 条 target absent）。
- 脚本：`../../scripts/extract_states.py`

```
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python runs/whitebox-20261107/scripts/extract_states.py \
  --checkpoint <ckpt> --run-dir <run> --out runs/whitebox-20261107/tasks/03-states/states-200.npz --limit 200 --seed 0
```

`--limit N` 用 `--seed` 抽样，target absent / present 各占一半；`--ids-file` 按给定 id 抽取；两者都不给时跑全量。

## 输入构造（与训练和推理一致）

- 音频通过 Catalog 读取，转成 int16，做法同 `spec_check.load_ref`。AmphionEval 发送的也是 PCM_16 WAV。enrollment 裁剪或补零到 5 s（服务约定），mixture 保持原长。之后经 `prepare_ts_clips` 补齐到 hop 整数倍，和训练相同。
- 每条样本组成一行：`{"messages": [system=context, user="<audio>", assistant=raw_text], "audios": [enroll], "mix_wav": mix}`，交给 ms-swift template（`Qwen3ASRTemplate._encode`，train mode）处理。enrollment 和 mixture 分别提取 Mel，中间插入学习得到的 SEP。
- 用 vLLM 输出的 `raw_text` 做 teacher forcing。vLLM 是从 `<|im_start|>assistant\n` 之后开始生成的：`completion_tokens` 等于重新分词后的 token 数 + EOS。重新分词得到的 token 序列与 vLLM 一致的比例为 99.5%（200 条）；不一致的只出现在 `<asr_text>` 之后的转写文本里。
- **决策位置**：assistant 回合的第一个 token `language`（id 11528）。这个位置的 logits 预测语言名 token。`language None` 分词为 `['language', ' None'(2240), '<asr_text>']`，`' Chinese'` = 8453，`' English'` = 6364，三者都是单个 token。

## 注意力实现（重要）

- LLM 用 eager attention。qwen_asr 的 `_can_record_outputs` 写的是 `Qwen3ASRThinkerTextAttention`，但实际的层是 `Qwen3ASRTextAttention`，所以 `output_attentions=True` 返回的是 None。attention 权重改为在每层 `self_attn` 上挂 hook 获取，只保留决策位置那一行。
- Audio tower 用 SDPA 且不加 mask，即在 [enroll, SEP, mix] 上做一个全局窗口。训练时就是这样（encoder 只在 FA2 下才会用 `cu_seqlens` 分窗），TS-ASR vLLM 服务也是这样（`tsasr/global_audio_attn.py` 把 `n_window_infer` 设为 1e9）。**如果改用 FA2，会按 config 的 800 帧分窗**：8 条 smoke 样本中首 token 一致率降到 4/8，4 条负样本里有 3 条从 None 变成输出文本，margin 最大约 16 nat。
- 验证用的 `output_hidden_states` 有 29 项，前 28 项与 hook 得到的值相同；最后一项是**经过 final norm 之后**的值。本文件中 `residual[:, 28]` 是 norm 之前的值，logit lens 会先做 norm 再过 lm_head。

## 验证结果

| 集合 | 样本数 | 首 token 一致率 | 正样本 | 负样本 | 负样本 None/非 None（vLLM→HF） | placeholder 数不一致 |
|---|---|---|---|---|---|---|
| states-200（seed 0，均衡） | 200 | 1.000 | 1.000 | 1.000 | none→none 99，text→text 1，交叉 0 | 0 |
| states-full（全量） | 26,750 | 0.9988 | 0.9992 | 0.9984 | none→none 13,263，text→text 512，none→text 6，text→none 16 | 0 |

- placeholder 检查（每条样本）：`<|audio_pad|>` 数 = `audio_token_count(T, True, enroll_n_frames)` = audio tower 输出长度，且 `enroll_n_frames` = 500（5 s）。enrollment 固定为 65 个 token，之后是 1 个 SEP，再之后是 mixture。
- 每条样本上，最后一层的 logit lens argmax 都等于模型 logits 的 argmax；`prediction` 为空当且仅当输出的语言 token 是 None。
- 全量中有 33 条不一致，全部是近似平局：HF 最后一层 None 与 Chinese 的 log-prob 差中位数为 0.0，最大 0.5 nat。这是 bf16 eager 与 vLLM kernel 的数值差异，不是输入构造的问题。分析时可以用 `final_argmax != emitted_token` 把它们排除。
- 全量标签分布：target absent 且输出文本（false alarm）528 条，absent 且输出为空 13,269 条，present 且输出为空（miss）292 条。
- 吞吐（1 张 A800，batch 1）：200 条时为 12.1 samples/s（8 个 worker，含预热）；全量为 21.4 samples/s（12 个 worker，1,252 s）。全量 npz 为 7.5 GB。

## 输出 schema（`states-*.npz`，S = 样本数）

| key | shape / dtype | 含义 |
|---|---|---|
| `id`, `source`, `split`, `reference`, `raw_text` | (S,) str | 元数据；`split` 例如 `abs-k1`、`enroll-clean-neg` |
| `reference_empty`, `prediction_empty` | (S,) bool | target absent = `reference_empty`；false alarm = `reference_empty & ~prediction_empty` |
| `residual` | (S, 29, 2048) f16 | 决策位置的 LLM 残差流：[0] 为 embedding 输出，[l] 为第 l 层输出（final norm 之前） |
| `audio_enroll_mean`, `audio_mix_mean` | (S, 25, 1024) f16 | audio tower 状态，分别在 enrollment / mixture token 上取平均：[0] 为 conv+PE 输出，[k] 为第 k 个 encoder 层输出（`ln_post`/proj 之前） |
| `audio_sep` | (S, 25, 1024) f16 | 每层 SEP token 的状态 |
| `attn_mass_heads` | (S, 28, 16, 5) f16 | 决策位置到各 bucket 的 attention 质量，按层、按 head |
| `attn_mass` | (S, 28, 5) f32 | 同上，对 head 取平均；bucket 顺序见 `attn_buckets` = enrollment, sep, mixture, system, other |
| `lens_logprob` | (S, 29, 3) f32 | logit lens 的 log-prob，对应 `candidate_ids` = [' None', ' Chinese', ' English'] |
| `lens_emitted_logprob` | (S, 29) f32 | 实际输出的语言 token 的 logit lens log-prob |
| `lens_top1` | (S, 29) i64 | 每层的 argmax token |
| `final_argmax`, `emitted_token` | (S,) i64 | HF 最后一层 argmax / vLLM 输出的语言 token |
| `decision_pos`, `seq_len`, `n_audio_pad`, `n_enroll_tokens`, `n_mix_tokens`, `enroll_n_frames`, `n_mel_frames` | (S,) i64 | 位置记录 |
| `retokenized_matches_vllm` | (S,) bool | 重新分词后的 assistant token 数 + EOS 是否等于 vLLM 的 `completion_tokens` |

Attention bucket 定义：`system` 是从第 0 个 token 到第一个 `<|im_end|>`（含）的 system 回合；`enrollment`/`sep`/`mixture` 是 audio pad 的三段；`other` 是剩余部分（user/assistant 标记、`<|audio_start|>`/`<|audio_end|>`、决策 token 自身）。

Audio 帧到片段的对应：packed 的 tower 序列 = `conv_token_count(enroll_n_frames)` 个 enrollment token，然后 1 个 SEP，然后 `conv_token_count(mix_frames)` 个 mixture token（`sep_token.stitch_sep_between_segments`）。encoder 层不改变长度，所以同样的切片适用于所有层。

每个 npz 旁边的 `*.summary.json` 记录全部验证数字；`*.log` 是运行日志。
