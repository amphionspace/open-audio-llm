# 真实会议与长音频大规模 SOT 训练

> 历史计划与当时运行记录。2026-09-22–29 的实际进展及结果见[近期实验记录](../experiments/2026-09-22-to-29.md)；后续新推理和评测统一使用 vLLM。

更新：2026-09-21。当前恢复运行：`qwen3-asr-sot-meeting-long-recovery-20260921`；原 v2 因显存不足退出，恢复说明见文末。

按用户最新要求直接训练 60,000 步，encoder、aligner、LLM 全部更新，不安排试训。目标是改善真实会议的转写、说话人分离和时间戳，增加长音频覆盖。普通 ASR 继续参与既有优化目标，其退化不作为阻断或早停条件。

[W&B](https://wandb.ai/1016097967-amphion/open-audio-llm/runs/qwen3-asr-sot-meeting-long-recovery-20260921-r2) 在训练前创建；实际状态以 [plan.json](../../runs/qwen3-asr-sot-meeting-long-recovery-20260921/plan.json)、日志和远端真实 step 为准。此前 MOSS 对比与自动评分任务保持停止。

## MOSS 参考与短板

参考 [论文 v7 §3](https://arxiv.org/pdf/2601.01554v7) 和 [官方实现](https://github.com/OpenMOSS/MOSS-Transcribe-Diarize)。论文没有公开真实、合成和普通 ASR 数据的混合比例，本轮比例是我们的选择。

| 维度 | MOSS 公开方法 | 我们的差异与本轮处理 |
|---|---|---|
| 说话人数 | 合成选 2～12 人 | 旧合成 1～5 人；新增真实会议，AISHELL-4 窗口最多 7 人 |
| 发言轮次 | 原句按词切为连续片段，采样长度并交替排列 | 旧合成每人仅 1～3 个整句；真实长会议补充跨轮次身份一致性，旧合成尚未升级词级切分 |
| 重叠 | 采样间隔、限制重叠，使用低能量边界和短交叉淡化 | 旧合成压缩整段时间线，高重叠偏多；保留条件并降低 high/dense 权重 |
| 声学 | 加噪与 RIR，SNR 0～15 dB；真实会议远场通道取均值 | 时间戳合成没有这些增强；新增真实远场音频，原始远场通道全部取均值 |
| 长度 | 最长 90 分钟、128k 上下文，含音频块时间标记 | 旧训练最长 28 秒；本轮约 30/120/300/600 秒、16k 上下文，没有实现 MOSS 的长音频架构 |

此前共同评分仅 123 段、46.87 分钟：Qwen/MOSS cpCER 为 44.69%/20.19%，DER（±0.25 秒）为 25.27%/9.65%；83 个多人片段中人数正确的分别为 3/58。这支持加强会议监督，但不是完整测试集成绩，也不能证明所有错误都由数据造成。

## 数据与隔离

由 AmphionData 的 `amphiondata.multispeaker.meetings` 准备，派生版本固定为 `meeting-long-v1-20260920`，通过本轮独立 Catalog 接入。

| 来源 | 原版本 / 划分 | 原始音频 | 派生训练窗口 |
|---|---|---:|---:|
| AISHELL-4 | `icefall-20260908:train`，L/M/S | 189 场，106.41 小时 | 19,066 |
| AliMeeting 远场 | `icefall-20260908:train` | 209 场，111.36 小时 | 20,341 |
| AliMeeting 开发集 | `icefall-20260908:dev` | 8 场，4.21 小时 | 不进入训练 |
| 合成时间戳 SOT | `synthetic-v2-timed-trial-v1-20260920` | 已有对齐通过的混音 | 1,644,206 |
| 普通 ASR | 沿用上一轮固定来源、版本与排除规则 | 中文、英文回放 | 复用已有索引 |

217.77 小时是源音频独立时长；39,407 个窗口是四种尺度的重复视图，不是独立会议数。筛选后各视图有效小时数不同，见 [数据审计](../../runs/qwen3-asr-sot-meeting-long-20260920-v2/meeting-data-audit.json)。

切分不截断发言；不可分重叠簇允许超过目标窗口，训练硬上限 650 秒。保留原文、重叠与说话人，窗口内按首次发声编号，同一人跨轮次一致；目标为 `[S1][0.32-2.48] 原文`。时间相对窗口起点，真实会议来自人工发言边界，合成来自自动对齐。真实目标不再查询合成 alignment-index。

按 tokenizer 和保守音频 token 预算检查完整目标，159 个训练窗口因超过 16,384 上下文排除；空文本标注和不可分超长重叠簇另有排除记录，未截断目标或伪造文本。无发言标注区间保留。完整 AliMeeting dev 不筛难例，其中 3 个 600 秒窗口超出本轮上下文预算，保留目标并记录限制。

未找到适用的会议 clean 训练版本，使用现有固定版本，不伪造 clean 标记。已有 clean 回放保留 `require_clean_pass: true`。官方 train/dev/test 不混用，AISHELL-4 与 AliMeeting test 均不进训练；此轮没有另行证明官方划分之间所有说话人身份互斥。

时间戳记录继续禁用变速和 RIR，普通 ASR 保留已有增强。旧合成的词级切分、噪声/RIR、8～12 人覆盖仍是后续数据短板。

## 配比与正式配置

比例按抽到的样本数计算，不按音频小时数计算；长会议会占更高的音频时间和计算比例。

| 类别 | 比例 |
|---|---:|
| 合成时间戳 SOT | 40% |
| AISHELL-4 会议 | 20% |
| AliMeeting 会议 | 20% |
| 中文普通 ASR | 15% |
| 英文普通 ASR | 5% |

每套会议内部目标 30/120/300/600 秒的比例为 10%/25%/35%/30%。合成保留各语言、人数组总权重，组内 none/low/medium/high/dense 为 25%/35%/25%/10%/5%；单人组仍只有 none。

40% 真实会议让会议与旧合成获得相近的样本更新机会，20% 普通 ASR 持续提供识别目标。这是尚待验证的实验选择，不是 MOSS 配方。

| 参数 | 值 |
|---|---|
| 初始化 | 上一轮时间戳 `checkpoint-2000` |
| 更新模块 | encoder、aligner、LLM 全部 |
| 步数 / warmup | 60,000 / 500 |
| encoder / aligner / LLM 峰值 LR | `1e-5 / 2e-5 / 1e-5` |
| 优化器 | AdamW fused，weight decay 0.01，梯度裁剪 1.0，cosine |
| 精度 | FP32 主参数 + BF16 autocast，gradient checkpointing |
| GPU / 梯度累积 | 2 × A800 80GB / 4 |
| 上下文 | 16,384 tokens |
| 动态组批 | 最多 4 条，总音频预算 650 秒，8 个时长桶 |
| 目标函数 | 既有 sample-mean CE；普通 ASR teacher KL 权重 2 |
| teacher | 原始 Qwen3-ASR-1.7B，冻结 |
| 保存 | 每 1,000 步保存模型、优化器、scheduler、采样游标，保留最近 3 份 |

从旧权重初始化，新建优化器与 scheduler，不恢复旧 optimizer step。动态 batch 的实际样本数以日志为准。此轮不重跑 MOSS 或固定集推理，不设置 ASR 保持门槛；训练数值错误和进程失败仍须处理。

## 核验与记录

正式进程启动检查权重起点、精度、可训练模块、实际 LR 和数据版本；第 5/20 步检查代表性 encoder、aligner、LLM 权重确实更新，随后继续同一运行。这些是正式训练核验，没有单独试训阶段。

[运行目录](../../runs/qwen3-asr-sot-meeting-long-20260920-v2/) 保存冻结代码、依赖、清单 SHA256、配置、日志和核验结果。W&B 独立 CPU 进程同步 loss、ASR/SOT CE、KL、梯度、吞吐、显存和实际长度，训练退出后同步最终状态；凭据只从本机 shell 配置加载。

数据准备已执行 `PYTHONPATH=../AmphionData/src:../audio-data-contract/src /ai_sds_wuzz/MODELS/miniconda3/envs/amphionft/bin/python -m pytest ../AmphionData/tests/test_meeting_sot.py -q`，结果 4 passed。覆盖完整重叠簇、长间隔后身份一致、极短时间边界及无标注区间保留。启动文件通过 Bash/Python 语法检查；实际训练核验另存运行目录，不将尚未发生的检查写成通过。

启动时从性能日志发现旧模板将音频截为 30 秒。已修复完整特征提取和 token 数量；初次启动的 15 步日志单独保留并标记排除，v2 从原始 checkpoint-2000 重新初始化，不继承这些更新。

长音频修复验证：`tests/test_audio_template_batching.py` 与 `tests/test_qwen3_asr_audio_batching.py` 合计 18 passed、1 skipped；验证完整特征、占位 token 数量、短音频特征兼容及 encoder 梯度与样本隔离。

正式 v2 启动核验已通过：双卡第 5 步 encoder、aligner、LLM 的代表权重均非零更新；实际长音频约 600 秒，最长上下文 15,756 tokens。W&B 远端已收到真实 loss 和 step，详见运行目录的 `startup-verification.json` 与 `wandb-startup-verification.json`。这些仅证明正确启动，不代表会议识别质量已经改善。


## 2026-09-21 显存修复与恢复

v2 在约第 1,574 步反向传播发生 CUDA OOM；另一 rank 等待 7,200 秒后退出。最新完整恢复点为 checkpoint-1000，随后约 574 步未保存。保留失败运行和全部日志。

新运行 `qwen3-asr-sot-meeting-long-recovery-20260921` 从该 checkpoint 恢复模型、AdamW 状态、scheduler、随机状态和 Catalog 采样游标，累计目标仍为 60,000 步。数据配置逐字节一致，长音频、16k 上下文、全参数更新、CE/KL 目标和采样比例保持原设置。

修复在交叉熵计算中按 128 个 token 分块，反向按块重算 softmax，直接写入完整 logits 梯度，避免保存或拼接额外的完整 FP32 词表矩阵。无需新增依赖。FP32/BF16、不同回答长度、CE/KL 和梯度等回归测试共 12 passed。10,176 × 151,936 的 FP32 输入上，loss 与抽查梯度一致；该 loss 运算峰值显存从 23.04 GiB 降为 11.66 GiB，减少 11.37 GiB。该数字是独立 loss 验证，不是完整训练进程总显存。

同时启用 expandable_segments 分配策略；保存间隔从 1,000 改为 500 步，保留最近 3 份；DDP 超时从 7,200 改为 300 秒，缩短单卡故障后的等待。没有以截断长音频或跳过困难样本处理 OOM。

[恢复运行 W&B](https://wandb.ai/1016097967-amphion/open-audio-llm/runs/qwen3-asr-sot-meeting-long-recovery-20260921-r2) · [恢复记录](../../runs/qwen3-asr-sot-meeting-long-recovery-20260921/recovery.json) · [显存验证](../../runs/qwen3-asr-sot-meeting-long-recovery-20260921/verify_loss_memory.json)

恢复后双卡已通过第 1,020 步核验：optimizer/scheduler 接续正常，encoder、aligner、LLM 代表权重持续更新，有效保存频率为 500 步。完整 600 秒输入继续参与训练；截至核验单卡峰值 allocated 显存约 61.3/59.8 GiB。最终恢复轨迹使用 W&B `-r2`，与保存频率修正前的启动记录分开；远端 loss 与当前本地日志逐项核对，见 `recovery-verification.json` 和 `wandb-recovery-verification.json`。
