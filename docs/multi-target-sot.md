# 可配置目标说话人的日志转写

多目标注册已接入现有时间戳 SOT 训练。保留一个模型、一个输出协议，输入可携带 0～3 段注册音频。训练默认每次读取重新采样注册，验证使用冻结清单。本次只实现代码与 CPU 验证，没有启动训练或 GPU 推理。

## 输入与输出

| 注册数 | mode | 行为 |
|---|---|---|
| 0 | `all` | 现有 SOT，按首次发声编号 `S1…` |
| 1～3 | `all` | 已注册者用 `T1…TK`，其余按首次发声编号 `S1…` |
| 1～3 | `targets_only` | 只输出已注册者；无人出现时输出空正文 |

`T1…TK` 严格对应注册顺序；交换参考音频顺序只应改变 T 编号。S 编号只在当前窗口内有效。所有时间戳相对待转写音频起点，允许不同人重叠，同一人各行按时间排列。

两位目标、`targets_only` 的完整 system prompt：

```text
前 2 段音频为参考说话人，依次编号 T1 至 T2，最后一段为待转写音频。
仅转写参考说话人的发言，每次发言一行：[编号][开始秒-结束秒] 文本。
```

`all` 把第二句换成「转写所有人的发言，其他说话人按首次发声编号 S1、S2……。」并保留行格式说明。无需在 prompt 中解释注册时长、数据采样或模型实现。

```text
language Chinese<asr_text>[T2][0.32-2.48] 我先介绍一下。
[S1][1.80-3.10] 好的，请继续。
[T1][4.20-5.30] 我补充一点。
```

保留 Qwen3-ASR 的原生语言头；空结果为 `language None<asr_text>`。`parse_target_segments` 可把正文转换为 `speaker_id/start/end/text` 列表。

## 注册时长与 diversity

选择 **1～5 秒**，每个注册槽独立均匀采样；若候选只有 D 秒，则采样 `Uniform(1, min(5, D))`。不足 1 秒的候选不可用，不重复或补零凑时长。这是音频片段长度，包含自然停顿，并不等同于纯语音时长。

依据：[ECAPA-TDNN](https://arxiv.org/abs/2005.07143) 使用 2 秒训练裁剪；[ECAPA2](https://arxiv.org/abs/2401.08342) 的可变长度训练包含随机 1～5 秒裁剪；[IDLAB VoxSRC 2020](https://arxiv.org/abs/2010.11255) 探索更长的 6 秒微调片段。综合短注册鲁棒性与本项目音频上下文开销，选 1～5 秒作为首轮区间。**均匀分布是本项目的实验选择，不是论文证明的最优分布，也不照搬 ECAPA2 的混合比例。**

每次 sampler occurrence 重采样：

- K：在可行的 `1…min(3, 可用身份数)` 中均匀采样。
- 身份：不重复选择，默认每个槽位以 20% 概率请求不在混音中出现的人；候选不足时从剩余可行身份中选择。
- 顺序：随机打乱身份与 T 编号的绑定。
- 来源、长度、裁剪起点：每个目标分别采样，可选同一人不同原始录音。
- 模式：默认 `all/targets_only` 各 50%。

随机种子包含训练 seed、epoch、记录索引和 occurrence，重复抽样会变化，同一采样状态可复现。候选池决定实际覆盖：只有一个可用身份时不能制造三人注册，没有缺席候选时不能保证 20% 缺席率。

沿用普通 ASR 20%、SOT 80% 的总体配比时，在 SOT 来源设置 `probability: 0.5`，候选充分时预期得到：普通 ASR 20%、无注册 SOT 40%、条件 all 20%、条件 targets_only 20%。缺少可用注册的记录回到原始完整 SOT，不丢掉原文。

`performance-rank*.jsonl` 记录实际注册样本数、K、mode、缺席人数和 1～2／2～3／3～4／4～5 秒直方图。实际分布应以这些计数为准。时间戳样本继续禁用改变时间轴的在线变速和 RIR；保留现有加噪与 SpecAugment。

## 数据契约与准备

输入必须是完整的 `speaker_attributed_asr` AudioRecord，`metadata.sot_output_format` 为 `aligned_utterance_timestamps_v1`，包含一个 `mixture` 音频槽。目标文本仍是原有 `[S1][开始-结束] 文本`。新增注释 JSONL 每条对应一个记录 ID，示意：

```json
{
  "id": "meeting-window-001",
  "partition": "train",
  "speaker_identities": {"S1": "corpus:alice", "S2": "corpus:bob"},
  "source_spans": [
    {"dataset_id": "meetings", "version": "v1", "split": "train", "cut_id": "session-01", "start": 20, "duration": 30}
  ],
  "enrollment_candidates": [
    {
      "speaker_id": "corpus:alice", "partition": "train", "single_speaker": true,
      "ref": {"dataset_id": "voices", "version": "v1", "split": "train", "cut_id": "alice-02", "start": 0, "duration": 8}
    },
    {
      "speaker_id": "corpus:absent", "partition": "train", "single_speaker": true,
      "ref": {"dataset_id": "voices", "version": "v1", "split": "train", "cut_id": "absent-07", "start": 0, "duration": 6}
    }
  ]
}
```

上游应提供带命名空间的真实身份、确认单人发声的区间、统一 train/dev/test partition、原始源片段。合成混音的 `source_spans` 列出所有原始成分；候选不能与其中任何区间重叠。不同别名的同一录音须在上游统一原始 cut 标识，本层不做全库音频指纹去重。通道通过 AudioRef 的 `channel` 明确指定。

适用 clean 训练版本优先且固定版本，source 设置 `require_clean_pass: true`；clean 混音的候选还需上游实际提供 `clean_pass: true`。没有适用 clean 版本可用现有训练集。连接工具不生成通过标记、不猜测文件名中的身份、不修改转写和时间戳。

从本 worktree 设置 `PYTHONPATH` 后运行，所有输出使用新路径：

```bash
python -m open_audio_llm.data.prepare_target_sot \
  --records /data/sot/train.jsonl --annotations /data/sot/enrollment-annotations.jsonl \
  --output /data/target-sot/train.jsonl

python -m open_audio_llm.data.prepare_target_sot \
  --records /data/target-sot/dev-annotated.jsonl \
  --freeze-seconds 3 --mode all --seed 42 \
  --output /data/target-sot/dev-3s-all.jsonl
```

对同一开发清单、seed 分别生成 1／2／3／5 秒、两种 mode 面板，使用相同身份、顺序、来源、起点。该时长对照需要候选支持 5 秒；不足时命令报错，不静默过滤。它是额外对照面板，保留完整原始开发集；短候选可另行冻结合法注册视图并报告覆盖率。

将新清单作为 **新 DatasetSpec 版本的 records_artifact** 登记到独立 Catalog；引用的原音频 DatasetSpec 和 audio-index 仍需可解析。不要替换当前训练使用的 manifest、Catalog 或版本。`fixed_enrollment` 写在原始完整 SOT 记录 metadata 内，验证读取时再确定性地派生 T/S 标签。

## 配置与训练入口

沿用现有 Catalog YAML，在已经注释好的 SOT source 增加：

```yaml
enrollment:
  probability: 0.5
  min_seconds: 1.0
  max_seconds: 5.0
  max_targets: 3
  absent_probability: 0.2
  mode: random
```

`enrollment: {}` 使用上述默认值。只可选取 1～5 秒内的子区间。普通 ASR source 不加此字段；训练配置仍需遵守全部 source 同时设置 weight 的现有规则。验证 source 指向冻结清单并添加 `enrollment: {}`；缺少 `fixed_enrollment` 会报错。batching 的时长预算自动增加最多三段注册及 SEP 的余量。

环境边界：`qwen-asr==0.0.6`、SDPA；CPU 检查使用 `ms-swift==4.5.3`。复用已有依赖，不修改共享训练环境。条件 SOT 支持本项目 Qwen3-ASR SFT，关闭 `audio_encoder_parallel`，保留梯度检查点。新入口继承现有 SOT 的学习率和全量更新设置，最大上下文设为 16,384；超长样本在 Catalog 编码时抛错，不截去音频或目标尾部，须在数据准备阶段按合法边界切窗。

模型复用 `/222042021/lx/AmphionASR-1.7B-v3` 多目标实验后续 v4 的「各段独立卷积 + SEP + 联合音频编码」组织方式，明确传递可变段长；不复制共享环境补丁。处理顺序是 `[e1, SEP, e2, SEP, …, mixture]`，LLM 侧使用一个完整音频占位区间。无注册样本保留原编码路径。以本项目 SOT checkpoint 初始化并新增可训练 SEP，**不是把 v3/v4 与 SOT 权重相加**。checkpoint 保存 `target_sot_audio` 和 SEP，重载时恢复。

仅在为新实验分配资源后执行，配置是参数唯一来源：

```bash
open-audio-llm train --config examples/configs/train/target-sot.yaml --dry-run
open-audio-llm train --config examples/configs/train/target-sot.yaml
```

先在 YAML 中改成冻结的 SOT checkpoint、原始 Qwen3-ASR teacher、带注册注释的数据配置和实际分配的 GPU。启动器负责 W&B 记录与远端核验，checkpoint 按 `storage` 同步到对象存储。

分支原实现把 replay 目标改为按监督 token 平均；`main` 保持按样本平均。需要按 token 平均时，在数据配置写 `objective.reduction: token_mean`（不能与 `separate_prefix` 同用）。

## vLLM 推理与评测

```python
from open_audio_llm.integrations.vllm.target_sot import TargetSOTVLLM

model = TargetSOTVLLM("/models/target-sot-checkpoint", max_model_len=16384)
result = model.transcribe("meeting.wav", ["alice-2s.wav", "bob-4s.wav"], "all")
print(result["prediction"])
```

默认在 CPU 仅加载 checkpoint 音频塔，复用训练的独立分段编码，再将音频 embedding 交给 vLLM 解码。文本生成只用 vLLM，不加载 Transformers 文本模型作回退。入口要求完整本地 safetensors checkpoint；适配现有 vLLM 0.17.0／0.18.0 接口，当前环境验证了 0.18.0 适配器导入与 `enable_mm_embeds` 参数。**尚未执行 GPU vLLM 端到端验证，也没有训练后质量结论。**

评测（请求构造、打分、分桶指标、W&B 记录）按仓库边界属于 AmphionEval，见 [使用 AmphionEval 评测](amphion_eval.md)。目标 SOT 的打分规则尚未迁入 AmphionEval：固定 `T1…TK` 身份，仅允许匿名 S 之间最优匹配，分别报告目标错误、目标漏词、缺席目标误触发、非目标输出、格式和句级边界指标，按语言、K、mode、注册时长分桶。原实现保留在 `feat/ts-diarization` 分支（`src/open_audio_llm/eval/target_sot.py`），迁移前不删除该分支。

## 当前验证范围

CPU 测试覆盖动态且可复现的采样、候选血缘和 split、目标编号置换、缺席目标、真实 Catalog 裁剪与缓存、固定验证视图、Swift 模板与混合组批、长混音特征保留、SEP 梯度、checkpoint 往返，以及 vLLM 请求使用相同音频 embedding。另检查原音频 batching、Catalog、retention 和性能计数相关回归。

正式训练前仍需发布带真实身份／注册候选的来源清单，冻结初始化 checkpoint，并在新分配资源上验证完整 vLLM 推理。本文不把测试夹具当作已就绪的正式训练数据。
