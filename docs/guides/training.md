# 训练入口与复现

训练直接从 Catalog 在线构造数据，运行参数集中在 [YAML 配置](../../examples/configs/README.md)。先填写模型、Python、GPU 和数据路径，预览确认后启动：

```bash
open-audio-llm train --config examples/configs/train/sft.yaml --dry-run
open-audio-llm train --config examples/configs/train/sft.yaml
open-audio-llm train --config examples/configs/train/grpo.yaml
```

SFT 的冻结范围通过 `task.arguments` 中的 `--freeze_vit`、`--freeze_aligner`、`--freeze_llm` 配置。阶段切换先按 [模型配置](../../examples/configs/model/merge-lora.yaml) 合并 adapter，再将下一阶段模型路径写入配置。

GRPO 使用普通 Catalog 配方，不支持 SFT 的 samples/reps、动态 batch 或 SpecAugment；示例显式使用 vLLM server。先启动 [rollout 配置](../../examples/configs/rollout/vllm.yaml)，两者使用不同 GPU。验证全局 batch 必须能被 `--num_generations` 整除。

```bash
open-audio-llm rollout --config examples/configs/rollout/vllm.yaml
open-audio-llm model --config examples/configs/model/merge-lora.yaml
```

### TS-ASR 提示

TS-ASR 样本的 system 来自 `tsasr/ts_prompt.py` 的 `TS_SYSTEM`（"Transcribe only the enrolled speaker. Output nothing if they do not speak."，2026-11-06 起），有热词时另起一行 `Hotwords: ...`。注册长度不写进提示，由请求的 `enroll_seconds` 决定。

### 成对相似嗓音损失（SFT）

数据配置写 `objective.paired_lookalike: {weight, prob, top: 4, margin: 2.0, gap: 20.0}` 后：
- **生成孪生题**：在线混音的目标在场样本以 `prob` 的概率得到一条"孪生注册"。它取自相似嗓音表里与原注册最近的前 `top` 个同语料说话人，按原注册的方式处理，并截成与原注册相同的长度。
- **计算成对损失**：训练时把这些样本的注册 Mel 换成孪生注册，再前向一次，只读决定出不出字的那个 token。损失要求同一混合段在孪生注册下输出 ` None`、在真注册下出字（两边各留 `margin` 的余量），并且孪生的 None 领先量比真注册高出 `gap`。达到后不再继续推开，避免把真注册的 logit 推到失控（`integrations/ms_swift/paired_lookalike.py`）。

损失按监督 token 数归一，`weight` 为 1 时，每对相当于两边各多一个监督 token。

前提条件：
- 要求 SFT 开启 `--supervised_head_loss`，不能开 `packed_text`。可以与联合 MTP 同时使用，从带 MTP 的 checkpoint 继续训练时用 `--joint_mtp_init` 加载它的 `mtp/`。
- 干扰源里要登记同语料的相似嗓音表，例如 `/workspace/data/lookalike/v2-samecorpus-20261108/`，做法见 `runs/whitebox-20261107/tasks/11-lookalike`。

训练日志里的 `pair_twin_none` 和 `pair_present_speaks`，分别是孪生题输出 None、真注册出字的比例。依据见白盒 task 12。

### Qwen3-ASR GRPO

Qwen3-ASR（`--model_type amphion_asr_1.7b`）用 `--reward_funcs qwen3_asr_accuracy qwen3_asr_presence qwen3_asr_hotword qwen3_asr_format`，奖励按模型原生输出 `language X<asr_text>文本` 计算。数据配置里有三项相关设置：

- `grpo` 一节：按 SFT 回放配方的来源比例抽 `prompts` 条 prompt。
  - `mined` 列出挖掘计划，格式为 `{plan, share}`，由 `python -m open_audio_llm.scripts.grpo_mining` 生成，按 `share` 混入。
  - `mined_only` 中的来源只经挖掘计划进入训练。
  - 开启 `--log_completions` 时，`completions.jsonl` 和 W&B 表多一列 `grpo_meta`（JSON 字符串），记录每条 prompt 的来历，供事后按计划、来源或混音方式切分 rollout：
    - `origin`：`replay` 或挖掘计划（`<任务目录>/<计划名>`）；
    - `source`、`record_id`、`draw`：来源、记录和抽样三元组，可用 `draw` 复现同一段音频；
    - `task`、`target_empty`：任务和参考是否为空；
    - 在线混音时另有 `enrollment`（时长、污染用的片段）和 `mixture`。`mixture` 包括类型（`absent`/`present`/`nonspeech`/`scene-*`）、时长、前后留白、干扰片段及其是否取自相似说话人表（`lookalike`）、其他人说话总时长。
- `objective.presence_contrast: true`：一组采样全部判错“该不该出字”时，组内最后一条换成参考答案。否则这组优势全为 0，得不到梯度。
- `rollout: {engine: vllm, gpu_memory_utilization: 0.25, max_model_len: 4096}`：rollout 改由训练进程内的 vLLM 解码。
  - 策略模型自己的音频塔按训练模板算出音频 embedding，vLLM（`Qwen3ASRTargetSOTForVLLM`）只解码文本；每次优化器步后同步文本权重。
  - 训练环境须装 `vllm==0.18.0`，torch 和 Transformers 版本与训练环境一致。
  - `--use_vllm` 保持 `false`。
  - 建议加 `--log_rollout_offpolicy_metrics true`，记录 rollout 与策略的偏离。
  - 默认的 transformers rollout 在 Qwen3-ASR 上以 `use_cache=False` 生成，因为模型配置里 `text_config.use_cache=false`。
  - 2026-11-02 的检查：
    - 同一权重下贪心输出 77/80 条与 transformers 逐 token 相同，不同的 3 条都是 TS 长输出。
    - 采样 token 的 logprob 与策略 teacher forcing 平均绝对差 0.003。
    - 4 卡训练每步从约 75 s 降到 7–10 s。

### 联合训练 MTP 的投机解码

`--joint_mtp_weight` 训出的 checkpoint 带 `mtp/`，它本身就是 vLLM 0.18 的草稿目录（`save_mtp` 写入一层 Qwen3 的配置，架构 `EagleQwen3ASRMTP`，类在 `src/open_audio_llm/tsasr/vllm_mtp.py`，插件注册）。旧 checkpoint 先用 `write_vllm_config` 补配置。服务加：

```
--speculative-config '{"method": "eagle", "model": "<checkpoint>/mtp", "num_speculative_tokens": 2}'
```

走 EAGLE 提议器而不是 vLLM 的 `mtp` 方法：后者只认 `MTPModelTypes` 和 Qwen3-Next 的层。embed_tokens 与 lm_head 共享目标模型的，草稿只用文本 embedding，与训练一致。v16 avg3 实测（`runs/joint-mtp-spec-20261107/`）：k3 每次校验接受 3.62 个 token，单路普通 ASR 均值 103→63 ms，推荐 k2。

### 投机解码草稿头（MTP）

`--draft_head_steps N`（N>0）冻结 Qwen3-ASR，只训一层 EAGLE 草稿头，供 vLLM 投机解码使用。草稿头按 N 步展开，第 k 步的损失权重为 `--draft_head_step_decay` 的 k−1 次方（默认 0.8）。只支持 SFT 数据配置，不能与 `supervised_head_loss` 或数据 objective 同用。

- 架构 `--draft_head_arch`：
  - `eagle1` 读最终 norm 输出，对应 vLLM 的 `eagle`。
  - `eagle3` 融合第 2 / 14 / 25 层隐状态，对应 vLLM 的 `eagle3`。
- 训练目标 `--draft_head_targets`：
  - `reference` 用数据标注。
  - `greedy` 先用冻结模型贪心重写目标，每步慢数倍。
- 从已有草稿头续训：`--draft_head_init <model.safetensors>`。
- 产物：每个 checkpoint 下的 `eagle/` 是 vLLM 草稿目录。服务加 `--speculative-config '{"method": "eagle3", "model": "<checkpoint>/eagle", "num_speculative_tokens": 3}'`。
- 校验是贪心的，主模型输出不变。草稿头读的是骨干隐状态，换骨干要重训，否则接受率下降。
- 实验记录：`runs/mtp-draft-head-20261031/`。
  - 选定 EAGLE-3、k=3。
  - 10000 步草稿头：普通 ASR 单路均值 100→67 ms，TS 83→64 ms。

新执行自动分配 `attempts/编号`。恢复失败执行必须显式指定已有记录，且任务声明支持恢复；完成记录不会默认重跑。训练的模型/优化器恢复路径写在 YAML 的 `--resume_from_checkpoint` 中，不接受环境或 CLI 覆盖。只保存模型时，须作为新训练初始化，不能宣称恢复优化器。

每次训练和评测同步启动 CPU W&B 记录器。只有远端 run URL 和启动指标核验通过后，才启动任务；退出后再次核验上传指标和远端结束状态。默认 entity 为 `1016097967-amphion`，project 为 `open-audio-llm`。

历史 2026-09-11 GRPO 检查使用 Python 3.11.15、torch 2.10.0+cu128、Transformers 4.57.6、ms-swift 4.1.1、TRL 0.29.1、DeepSpeed 0.18.9、PEFT 0.18.1。单卡 ZeRO-2 LoRA 完成 2 步，loss 为 0.707788 / 0.710823；168 个 adapter 张量发生变化。历史本地生成验证不代表新的 vLLM rollout 已完成端到端验证，也不代表识别能力提升。

详细采样和恢复语义见 [在线训练数据](online-training-data.md)，实验组织见 [实验规范](experiments.md)。
