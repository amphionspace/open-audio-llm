# TS-ASR 与普通 ASR 回放

此配方复用 `/222042021/lx/AmphionASR-1.7B` 的 TS-ASR 输入和标签协议，
通过固定配额、持续回放和逐域中文 CER 验收降低遗忘风险。
它是待训练验证的保守起点，不能仅凭数据比例保证能力不退化。

原项目预训练模型到 v9/68119 的 WenetSpeech test-meeting CER 从 5.853%
升到 13.388%，KeSpeech 从 4.568% 升到 6.909%，但 AISHELL 从 1.599%
降到 0.646%。依据为原项目 `exp/eval_vllm/` 中
`pretrained_qwen3_asr_1.7b_asr_full-2/summary.json` 与
`sft_v9_ckpt68119_asr_full-2/summary.json`。单看 AISHELL 或混合 loss 会漏掉
领域退化；现有记录不能分离数据、提示、标签和参数更新各自的因果贡献。

## 数据与采样

[qwen3_asr_ts_replay.yaml](../../configs/data/qwen3_asr_ts_replay.yaml) 直接消费
`audio-data-contract` 已登记的 LibriMix、AISHELLMix 训练集与测试集，不生成
ShareGPT，不修改上游音频或 Catalog。

| 类别 | 每 200 条的配额 | 各来源占总样本比例 |
|---|---:|---|
| 中文普通 ASR | 100 | WenetSpeech 20%、AISHELL 10%、AISHELL2 10%、KeSpeech 5%、AISHELL4 5% |
| 英文普通 ASR | 40 | LibriSpeech 10%、GigaSpeech 10% |
| 中文 TS-ASR | 30 | AISHELLMix 2spk 7%、3spk 6%、负例 2% |
| 英文 TS-ASR | 30 | LibriMix 2spk 7%、3spk 6%、负例 2% |

`replay.epoch_samples=20000` 定义逻辑 epoch，每个 200 条窗口都落实配额。
来源内部独立打乱，无放回遍历所有合格记录后才重新打乱回放；游标跨逻辑 epoch
延续。大语料不会挤掉中文回放，小语料不会耗尽退出。去重以记录为单位，
不消除不同混音记录共享的原始语音。

分桶后再次打乱完整 batch，减少长 TS 和短普通 ASR 连续成块。
配额保证针对全局完整采样窗口，不保证每卡、每个 batch 或每次梯度更新的比例。
DDP 尾部补批会略微改变整轮计数。比例按样本数计算，**不等于音频秒数、目标
token 数或梯度占比**，长 TS 仍可能贡献更多监督，因此起点将 TS 总量限制为 30%。

启动脚本先用 `METADATA_WORKERS=4` 个 CPU 进程建立可复用的磁盘索引，默认放在
`runs/catalog-metadata-cache`，可用 `AUDIO_DATA_METADATA_CACHE` 指定位置。
首次需要扫描原始 manifests；后续各 rank 和 DataLoader worker 按需读取记录，
时长和槽位使用紧凑数组，不再常驻数千万个 Python 记录对象。索引按来源定义、
文件路径/大小/修改时间、采样率和变速设置区分；共享目录使用文件锁，完成后才
供其他进程读取，不修改上游音频或 Catalog。

来源内的排列保留原有 seed 和 shuffle 算法，写成共享数组后跨逻辑 epoch 复用，
遍历完来源才产生下一轮排列。来源 `max_samples` 固定取前 N 条，仅用于快速检查，
不能替代正式运行的全语料随机回放。

## 输入与训练参数

普通 ASR 的 system 为空；TS system 明确指定目标说话人任务。训练和评估共用
相同拼接：enrollment 前 3 秒，不足补零，随后是 3 秒静音和完整 mixture，最终
只有一个 `<audio>`。中文答案为 `language Chinese<asr_text>文本`，英文使用
`English`，目标说话人不存在时为 `language None<asr_text>`。

波形增强后再拼接，参考段与静音始终各占 3 秒。变速概率 25%，SpecAugment
概率 15%，不加噪或混响。主音频最多 20 秒、最慢速度 0.9，TS 最长
`20 / 0.9 + 6 ≈ 28.23` 秒，为 30 秒特征窗口留出余量；组批也预算这 6 秒前缀。

从原生预训练模型新建 LoRA，冻结音频编码器和连接层，只训练语言模型线性层的
LoRA；rank=64、alpha=128、学习率 `2e-5`、cosine、5% warmup。
冻结编码器减少参数漂移范围，但不能代替回放与评估。默认 2000 步，每 250 步
保存并计算 dev loss，保留 8 个 checkpoint。延长训练时继续回放，不在后期切成纯 TS。

`batching.batch_merge=2`、`merge_window=4` 在原有分桶和 DDP 分配完成后，
将每次更新的四个原始 batch 按音频长度排序，配成两个合并 batch，减少 padding。
单个合并 batch 的上限为 16 条 / 240 秒，梯度累积从 4 改为 2。每次更新的完整
样本集合得以保留；组内顺序、浮点归约及 dropout 调用会变化，续训不保证逐位复现。
DataLoader 使用每卡 4 个常驻 worker，避免每个逻辑 epoch 重建进程。

`--audio_encoder_parallel true` 使用两个 CUDA stream 并行执行冻结的音频编码器，
保留每条音频原有的卷积、矩阵乘法和 attention 形状。音频长度一次传到 CPU，
避免逐条反复同步 GPU。调用方的 autocast 和 inference 状态传入工作线程，输出
恢复原有顺序后才进入语言模型。此路径限定 `qwen-asr==0.0.6`、SDPA、冻结且无
dropout 的编码器；不满足条件时明确报错，可传 `--audio_encoder_parallel false`
使用原生路径。默认不采用会改变 BF16 数值结果的全批量编码。

训练结束后，`gc-rank*.json` 记录训练阶段全量 Python GC 的实测停顿；
`performance-rank*.jsonl` 和 `gpu.csv` 用于比较吞吐、数据等待和 GPU 利用率。

checkpoint 中仍保存合并前的 batch 游标，支持旧版 `catalog_sampler.json` 恢复。
数据来源、增强设置、原始分桶参数和 world size 改变时仍拒绝恢复；磁盘索引开关和
`batch_merge`、`merge_window` 不改变数据身份。长度配对窗口必须整除一次梯度更新
包含的原始 batch 数；采样状态只在完整配对窗口结束或 epoch 结束时保存，不能保存
一半梯度累积的游标。用 `--resume_from_checkpoint /path/to/checkpoint-N`
恢复模型、优化器及数据进度，输出目录应另行指定以保留原实验。

## 启动与验收

沿用 [原生 Qwen3-ASR 环境与参数](README.md)，不新增依赖。
此入口使用已有 ms-swift 4.x 接入；具体版本与运行限制见该文档。

```bash
export MODEL=/path/to/pretrained/Qwen3-ASR-1.7B
export PYTHON=/path/to/environment/bin/python
export AUDIO_DATA_CONTRACT_ROOT=/path/to/audio-data-contract
export AUDIO_DATA_CATALOG=$AUDIO_DATA_CONTRACT_ROOT/catalog
export AUDIO_DATA_ROOTS_FILE=$AUDIO_DATA_CONTRACT_ROOT/roots.json
export PYTHONPATH=$PWD/src:$AUDIO_DATA_CONTRACT_ROOT/src
export CUDA_VISIBLE_DEVICES=2,3
export OUTPUT_DIR=$PWD/runs/qwen3-asr-ts-replay
export DATA_CONFIG=$PWD/examples/configs/data/qwen3_asr_ts_replay.yaml

# 固定开发集基线。
CUDA_VISIBLE_DEVICES=2 "$PYTHON" -m open_audio_llm.eval.ts_asr \
  --model "$MODEL" --data_config "$DATA_CONFIG" \
  --output_dir "$OUTPUT_DIR/base-dev"

bash examples/train/qwen3-asr/train_tsasr.sh

# 填写实际 checkpoint 路径，对候选分别验收。
CHECKPOINT=/path/to/checkpoint-250
CUDA_VISIBLE_DEVICES=2 "$PYTHON" -m open_audio_llm.eval.ts_asr \
  --model "$MODEL" --adapter "$CHECKPOINT" --data_config "$DATA_CONFIG" \
  --baseline "$OUTPUT_DIR/base-dev/summary.json" \
  --output_dir "$OUTPUT_DIR/candidate-dev"
```

评估加 `--audio_encoder_parallel` 可验证并行编码路径；基线和候选仍使用相同的
样本及 CER gate，不放宽阈值。默认评估保留原生编码路径。

评估默认读取 `validation`，固定最多 256 条/来源，不增强、不强制语言，最多生成
256 个 token。输出逐句预测、各域 CER/WER、正例漏识别率与 TS 负例误报率。
基线和候选的样本、参考文本、音频引用、采样率及解码配置必须相同，否则拒绝比较。
任一中文普通 ASR 来源 CER 高于基线，命令退出码为 1，结果仍保存到 `summary.json`。
默认容忍增幅为 0；`--max_cer_increase 0.005` 允许绝对增加 0.5 个百分点，
需明确接受这类退化后才调整。小子集会有波动，临界结果应扩大开发集验证。

这是独立的 checkpoint 验收命令，**不会自动中止 Trainer 或回滚权重**。
开发集未通过的 checkpoint 不应采纳；优先降低学习率或 TS 配额重新实验。
原项目 checkpoint 没有本项目采样器状态，不能直接 `resume_from_checkpoint`；
若作为模型初始化，中文保持基线仍应使用原生预训练模型，避免掩盖已有退化。

候选确定后，基础模型和候选分别加 `--split_group evaluation --samples_per_source 0`
做最终测试，以该次基础模型结果作为 `--baseline`。这会遍历所有符合配置过滤条件的
测试记录，默认只代表主音频不超过 20 秒的子集。覆盖更长录音需要另行确认原生
模型分段策略与 TS 拼接语义，不能直接去掉过滤便宣称是等价测试。
测试集不用于反复选择 checkpoint；中文 gate 通过后仍需检查英文与 TS 的独立成绩。

实现验证不等于完整训练结果；是否保住中文能力，必须用训练后的模型完成上述对照。

## 全参数 LLM 与基座回放约束

`train_tsasr_full.sh` 使用 `qwen3_asr_ts_full.yaml` 从原生基座全量更新 LLM，
包含 embedding、全部 decoder 层和 lm_head；audio encoder 及其投影继续冻结。
默认 12,000 步、学习率 `5e-6`、300 步 warmup、每 500 步保存/评估。
它使用现有 qwen-asr 0.0.6 / ms-swift 4.1.0 / Transformers 4.57.6 环境中的
DDP 和 AdamW，不支持此自定义损失与 DeepSpeed 或 logits_to_keep 同时启用。

旧配方负例只有总样本的 4%，且空转写仅有少量答案 token；按 token 平均时，
负例监督会进一步被长转写稀释。直接改成整条答案的样本均值也不够：真实试跑中，
短负例又让语言/存在性判断相对长正例获得过高权重，造成正例大量输出为空。
因此全参配方将 `language X<asr_text>` 前缀与正文（含 EOS）分别取 token 均值，
两者相加后，再在所有卡和梯度累积 microbatch 的样本之间平均。正负例的存在性
判断不会再随转写长度改变权重。中文普通 ASR 50%，英文
普通 ASR 10%，中英文 TS 各 20%；每种语言的双人/三人/负例分别占总量 5%/9%/6%。

普通 ASR 回放增加 `KL(base || student)`，系数为 10，同样分别平均前缀和正文，
只作用于有监督的答案位置。第一轮系数 1 的试跑仍未通过逐域中文 CER 门槛，
因此提高约束强度；该数值仍需开发集验证，不能视为零退化保证。
教师是 `RETENTION_TEACHER` 指定的冻结原生基座，默认等于 `MODEL`；不在 TS
样本上模仿基座。训练使用 FP32 主权重和 AdamW 状态、BF16 autocast 运算，
避免小学习率更新直接被 BF16 权重舍入。只投影有监督位置以控制词表 logits 显存。
这些措施降低退化风险，不能替代逐域 CER 验收。

TS 开发样本按注册记录 `source_record_id` 中首条原始语音对应的混音族留出 1%，
同族的双人/三人和正负目标一并划分。该划分属于同源混音族留出，**不是说话人
隔离**；说话人和部分原始语音可以与其他训练混音或普通 ASR 回放重叠。
已有 test split 不参与此划分。训练中的 teacher-forced dev loss 每 TS 来源取
128 条用于及时观察；生成式 CER/WER/误报率应另外在完整留出来源中固定抽样。

```bash
CUDA_VISIBLE_DEVICES=0,1,2 NPROC_PER_NODE=3 \
  OUTPUT_DIR=$PWD/runs/qwen3-asr-ts-full \
  bash examples/train/qwen3-asr/train_tsasr_full.sh
```

普通 ASR 验收仍使用相同基座、相同开发样本、每个中文域零 CER 增幅；不得用更低
TS loss 抵消中文退化。全参 checkpoint 评估通过 `--model /path/to/checkpoint-N`
加载，不使用 `--adapter`。全参优化器和回放配方与旧 LoRA 运行不同，不能复用旧
LoRA 的 optimizer/sampler 状态直接续训。

生成评估显式对外层模型和 thinker 的 GenerationConfig 启用 KV cache、关闭随机采样。
全参训练的 gradient checkpointing 可能将 `text_config.use_cache=False` 写入 checkpoint；
原生 Qwen wrapper 实际调用的是 thinker.generate，不能只修改外层 GenerationConfig。
推理必须统一此设置，避免反复计算整个音频前缀，以及由不同计算路径产生的比较偏差。
