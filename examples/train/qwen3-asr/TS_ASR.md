# TS-ASR 与普通 ASR 回放

此配方复用 `/222042021/lx/AmphionASR-1.7B` 的 TS-ASR 输入和标签协议，
通过固定配额、持续回放和逐域中文 CER 验收降低遗忘风险。
它是待训练验证的保守起点，不能仅凭数据比例保证能力不退化。

训练采用 clean 优先策略：有适用 clean 训练版本时使用，没有时允许原训练集。
AISHELLMix / LibriMix 当前版本保留原标注、未重新审核，仍可用于 TS 训练，
但不能描述为已清洗。未来取得适用 clean 版本后再显式替换，保持版本与采样可追溯。

原项目预训练模型到 v9/68119 的 WenetSpeech test-meeting CER 从 5.853%
升到 13.388%，KeSpeech 从 4.568% 升到 6.909%，但 AISHELL 从 1.599%
降到 0.646%。依据为原项目 `exp/eval_vllm/` 中
`pretrained_qwen3_asr_1.7b_asr_full-2/summary.json` 与
`sft_v9_ckpt68119_asr_full-2/summary.json`。单看 AISHELL 或混合 loss 会漏掉
领域退化；现有记录不能分离数据、提示、标签和参数更新各自的因果贡献。

## 数据与采样

[qwen3_asr_ts_replay.yaml](../../configs/data/qwen3_asr_ts_replay.yaml) 直接消费
`audio-data-contract` 登记的数据，使用下表配额；不生成 ShareGPT，不修改上游音频或 Catalog。

| 类别 | 每 200 条的配额 | 各来源占总样本比例 |
|---|---:|---|
| 中文普通 ASR | 100 | WenetSpeech clean v3 20%、clean W 5%、AISHELL 5%、AISHELL-2 5%、KeSpeech 10%、AISHELL-4 5% |
| 英文普通 ASR | 20 | LibriSpeech 5%、GigaSpeech 2%、Common Voice EN clean v1 3% |
| 中文 TS-ASR | 40 | AISHELLMix 2spk 7%、3spk 13% |
| 英文 TS-ASR | 40 | LibriMix 2spk 7%、3spk 13% |

`replay.epoch_samples=20000` 定义逻辑 epoch，每个 200 条窗口都落实配额。
来源内部独立打乱，无放回遍历所有合格记录后才重新打乱回放；游标跨逻辑 epoch
延续。大语料不会挤掉中文回放，小语料不会耗尽退出。去重以记录为单位，
不消除不同混音记录共享的原始语音。

分桶后再次打乱完整 batch，减少长 TS 和短普通 ASR 连续成块。
配额保证针对全局完整采样窗口，不保证每卡、每个 batch 或每次梯度更新的比例。
DDP 尾部补批会略微改变整轮计数。比例按样本数计算，**不等于音频秒数、目标
token 数或梯度占比**，长 TS 仍可能贡献更多监督，当前 TS 目标配额为 40%。

运行 YAML 的 `preflight` 先用 4 个 CPU 进程建立可复用索引，位置由数据 YAML 的
`metadata_cache` 指定，示例为 `runs/catalog-metadata-cache`。
首次需要扫描原始 manifests；后续各 rank 和 DataLoader worker 按需读取记录，
时长和槽位使用紧凑数组，不再常驻数千万个 Python 记录对象。索引按来源定义、
文件路径/大小/修改时间、采样率和变速设置区分；共享目录使用文件锁，完成后才
供其他进程读取，不修改上游音频或 Catalog。

来源内的排列保留原有 seed 和 shuffle 算法，写成共享数组后跨逻辑 epoch 复用，
遍历完来源才产生下一轮排列。来源 `max_samples` 固定取前 N 条，仅用于快速检查，
不能替代正式运行的全语料随机回放。

## 输入与训练参数

普通 ASR 的 system 为空；TS system 与 AmphionASR v3 相同。注册音取前 3 秒，
不足补零，混合音保持独立，中间不加静音。用户轮只有一个 `<audio>`，mixture
走 `mix_wav`。编码器对两段分别提 Mel、分别卷积，在进 audio transformer 前插入
可学习 SEP。中文答案为 `language Chinese<asr_text>文本`，英文使用 `English`，
目标说话人不存在时为 `language None<asr_text>`。

波形增强作用在各段上，参考段始终占 3 秒。变速概率 25%，SpecAugment
概率 15%，不加噪或混响。主音频最多 20 秒、最慢速度 0.9，TS 最长
`20 / 0.9 + 3 ≈ 25.23` 秒；组批预算这 3 秒注册前缀。训练不要设置
`AMPHION_TSASR_INSERT_SEP`。评测 concat 波形（注册 3 秒紧接混合音）使用
[独立服务配置](../../configs/serve/tsasr.yaml)，由启动器按 YAML 生成 SEP 与注意力设置。

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
恢复模型、优化器及数据进度，前提是 checkpoint 保存了完整训练状态；
输出目录应另行指定以保留原实验。仅保存模型时的限制见下文联合训练入口。

## 启动与验收

沿用 [原生 Qwen3-ASR 环境与参数](README.md)，不新增依赖。
以下命令只读配置文件；模型、GPU、batch、学习率和恢复 checkpoint 均写 YAML。
所有新 checkpoint 对比使用 vLLM；本页原有原生生成参数及结果描述作为历史协议保留，不再作为公开推理入口。
此入口使用已有 ms-swift 4.x 接入；具体版本与运行限制见该文档。

```bash
open-audio-llm serve --config examples/configs/serve/vllm.yaml
open-audio-llm eval --config examples/configs/eval/comparison.yaml
open-audio-llm train --config examples/configs/train/ts-replay.yaml
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
判断不会再随转写长度改变权重。当前由 AmphionRuntime 的 SpeakerVAD 承担目标
不在场时的拒识，本项目 TS 训练只使用目标在场的混叠正例。中文普通 ASR 50%，
英文普通 ASR 10%，中英文 TS 各 20%；每种语言的双人/三人分别占总量 7%/13%，
约保持原正例的 5:9 比例。原来的两组负例各 6% 已转给正例，TS 正例总量由 28%
提高至 40%；普通 ASR 回放比例保持不变。

普通中文回放使用 WenetSpeech clean v3 20%、clean W 5%、AISHELL 5%、AISHELL-2 5%、
KeSpeech 10%、AISHELL-4 5%；普通英文使用 LibriSpeech 5%、GigaSpeech 2%、
Common Voice EN clean v1 3%。三个 clean 来源显式设置 `require_clean_pass: true`，
按 Lhotse `custom.clean.pass` / AudioRecord `metadata.clean.pass` 为布尔 `true` 过滤；
失败或没有标记的条目不进入这些 clean 来源的索引。其余暂无适用 clean 训练版本的
来源继续使用原版本，不要求清洗标记。开发和测试标注沿用固定版本。
WenetSpeech clean v3 使用其
独立登记的清洗文本入口，不能只给原始 `icefall-20260908` 入口加过滤开关。
清洗来源中超出录音时间范围或时长非正的标注会在切分前剔除，避免无效标注
触发 Lhotse 切分断言；同一录音中的其他有效标注继续使用。

训练来源不设置 `samples` 或 `max_samples` 数量截断；20,000 条是一个采样
轮次的呈现量，采样器跨轮次继续遍历各来源的完整索引，不会每轮重复同一个
20,000 条子集。现有 0.5～20 秒主音频范围、增强及混音族留出划分继续生效。

普通 ASR 回放增加 `KL(base || student)`，系数为 10，同样分别平均前缀和正文，
只作用于有监督的答案位置。第一轮系数 1 的试跑仍未通过逐域中文 CER 门槛，
因此提高约束强度；该数值仍需开发集验证，不能视为零退化保证。
教师是 `--retention_teacher` 指定的冻结原生基座，默认等于 `--model`；不在 TS
样本上模仿基座。训练使用 FP32 主权重和 AdamW 状态、BF16 autocast 运算，
避免小学习率更新直接被 BF16 权重舍入。只投影有监督位置以控制词表 logits 显存。
这些措施降低退化风险，不能替代逐域 CER 验收。

TS 开发样本按注册记录 `source_record_id` 中首条原始语音对应的混音族留出 1%，
同族的双人/三人和正负目标一并划分。该划分属于同源混音族留出，**不是说话人
隔离**；说话人和部分原始语音可以与其他训练混音或普通 ASR 回放重叠。
已有 test split 不参与此划分。训练中的 teacher-forced dev loss 每 TS 来源取
128 条用于及时观察；生成式 CER/WER 应另外在完整留出来源中固定抽样。
当前全参配置的训练、开发和测试来源均移除了 TS 负例。TS 候选排序只比较中文
双人/三人 CER、英文双人/三人 WER，汇总时四项等权，负例 FAR 不参与分数或门槛。
正例空输出仍是漏识别，必须计入 CER/WER，并单独报告空输出率。中文逐域保持
门槛继续独立执行，不能用更低 TS 错误率抵消通用中文退化。

TS 输入是 3 秒 enrollment 和独立混音，中间用 SEP 分开，模型负责从混音中识别注册
说话人的文字。SpeakerVAD 的端到端拒识效果由 Runtime 验收；本项目的重叠识别
测试直接评估目标在场的混音，避免将前级误拒的难例从识别指标中排除。

切换到此配方时，以旧 checkpoint 的模型权重初始化新运行，并重新建立采样计划。
数据来源和权重已改变，旧 `catalog_sampler.json` 的签名不兼容，不能沿用旧采样
游标直接 `resume_from_checkpoint`。`--model` 指向旧 checkpoint 时，需显式将
`--retention_teacher` 指向原生预训练模型，继续约束相对原基座的中文退化。

```bash
open-audio-llm train --config examples/configs/train/ts-full.yaml
```

普通 ASR 验收仍使用相同基座、相同开发样本、每个中文域零 CER 增幅；不得用更低
TS loss 抵消中文退化。全参 checkpoint 评估通过 `--model /path/to/checkpoint-N`
加载，不使用 `--adapter`。全参优化器和回放配方与旧 LoRA 运行不同，不能复用旧
LoRA 的 optimizer/sampler 状态直接续训。

生成评估显式对外层模型和 thinker 的 GenerationConfig 启用 KV cache、关闭随机采样。
全参训练的 gradient checkpointing 可能将 `text_config.use_cache=False` 写入 checkpoint；
原生 Qwen wrapper 实际调用的是 thinker.generate，不能只修改外层 GenerationConfig。
推理必须统一此设置，避免反复计算整个音频前缀，以及由不同计算路径产生的比较偏差。

## Encoder、投影与 LLM 联合训练

`train_tsasr_joint.sh` 沿用全参回放目标，同时解冻 audio encoder、proj1/proj2 和
全部 LLM 参数。三个优化器参数组互不重复，包含 lm_head；学习率分别由
`--vit_lr`（默认 `1e-6`）、`--aligner_lr`（默认 `5e-6`）和 `--learning_rate`
（默认 `5e-6`）控制。默认 80,000 步，每 1,000 步保存模型并计算 dev loss，
保留最近 4 个 checkpoint；这些默认值仍需通过生成式评测验证。

```bash
open-audio-llm train --config examples/configs/train/ts-joint.yaml
```

联合入口关闭仅适用于冻结 encoder 的双 stream 路径，启用 encoder gradient
checkpointing。`--audio_encoder_batching true` 是可选的真正批处理路径：
保留每条音频内部 enrollment 与 mixture 的完整 attention，隔离不同样本并屏蔽
padding，支持 encoder 反向传播。它限定 `qwen-asr==0.0.6`、SDPA、100 帧卷积
块和零 dropout；与 `audio_encoder_parallel` 互斥，只作用于 student。
批处理会改变 BF16 浮点归约结果，不能保证与逐条编码逐位相同；评估时可显式
添加 `--audio_encoder_batching` 对照识别效果，默认评估仍使用原生路径。

联合入口默认 `--save_only_model true`，不保存优化器状态。
已有 checkpoint 可通过 `--model` 初始化新运行，但这会重新建立优化器，不能声称
精确恢复完整训练状态。加载本项目 FP32 checkpoint 时添加 `--torch_dtype float32`，
避免加载阶段先舍入到 BF16；`--retention_teacher` 仍指向原生基座。
如果需要完整状态续训，应在产生 checkpoint 前显式设置 `--save_only_model false`，
并保持数据配方和采样器签名兼容。

可选参数 `--retention_eval_script /path/to/evaluate_checkpoint.py`
与 `--retention_eval_interval 2000` 在首次保存、间隔步数对应的保存和末步保存时，
暂停所有 DDP rank，在 rank 0 的 GPU 上调用同一 Python 环境中的脚本。
脚本依次接收 checkpoint 路径与评测输出目录两个位置参数，负责加载模型和计算指标；
间隔只在保存事件中检查，应设为保存间隔的整数倍。训练模型仍驻留显存，需为评测
模型预留空间。日志写入 `retention-evaluations/checkpoint-N/runner.log`。
脚本退出码非零或运行超时会让所有 rank 报错停止；超时预算为 `ddp_timeout - 60`
秒，最少 1 秒。若希望仅记录 CER 门槛失败并继续训练，脚本应保存该结论后返回 0，
执行故障仍返回非零。此回调不会自动回滚权重。
