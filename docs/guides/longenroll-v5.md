# 长注册 TS-ASR（AmphionASR v5 模型）

`open_audio_llm.longenroll` 是 amphionasr-v5 长注册模型在本框架里的实现：Qwen3-ASR-1.7B 端到端训练，
注册音频 1 秒～5 分钟。数据用 [v5 数据配方](amphionasr-v5-data.md)，训练入口是
[longenroll-v5.yaml](../../examples/configs/train/longenroll-v5.yaml)（本机 8 卡）与
[longenroll-v5-acp.yaml](../../examples/configs/train/longenroll-v5-acp.yaml)（提交到 ACP cluster-whai，集群设置与
amphionasr-v4-acp.yaml 相同；checkpoint 暂留 `/workspace`，不同步对象存储），两者训练参数相同。模型超参在
[longenroll_v5.yaml](../../examples/configs/model/longenroll_v5.yaml)（与 amphionasr-v5 `configs/v5.yaml` 的 `model:` 相同）。

```bash
open-audio-llm train --config examples/configs/train/longenroll-v5-acp.yaml --dry-run
```

部署推理见 [推理（vLLM）](#推理vllm)。

## 模型

| 路径 | 做法 | 训练 |
|---|---|---|
| 注册 | 整段 mel 过 conv，再过原第 0–11 层的冻结副本，注意力限在 30 秒窗内；每窗压成 32 个槽，8 个全局 + 24 个局部查询读出 | 压缩器、查询训练；conv 与层副本冻结、不存激活 |
| 混合音 | conv（冻结）+ ECAPA 偏置，第 0–23 层全局注意力，第 12/16/20/22 层后接门控交叉注意力（GCA） | 第 0–23 层、LLM、GCA、ECAPA 投影训练 |
| ECAPA | wespeaker ECAPA-TDNN（16 kHz）。注册 ≤10 秒整段输入；更长的取开头，训练随机 5～10 秒、验证 10 秒 | 冻结 |

第一次 8 卡训练（`runs/longenroll-v5-acp` attempt 001）后加的 4 项修复，正式配置
[longenroll_v5.yaml](../../examples/configs/model/longenroll_v5.yaml) 全部打开（代码里默认关闭，旧配置与 checkpoint 不受影响），
诊断与验证记录在 `runs/longenroll-v5-dev-20261016` 的 05–07 任务：

| 配置 | 做法 | 原因 |
|---|---|---|
| `freeze_embeddings: true` | 冻结 `embed_tokens`（LM head 与之绑定，一起冻结），与 v3/v4 相同 | 训练它时范数 9 万步降到 49%、低频行方向偏移，ASR 训练 CE 在平稳数据上持续上升 |
| `enroll_encoder.feature_standardize` + `feature_stats` | 压缩前用训练集统计量（`longenroll/feature_stats.py`，974 条 TS 注册）按维度标准化冻结特征；统计量存为模型 buffer，随 checkpoint 与导出目录走 | 冻结特征按时间平均后说话人无关的公共分量占约 90% 能量，压缩后各说话人的注册记忆几乎相同（余弦 0.99），GCA 没有可用的说话人信息 |
| `gca.output_norm: true` | GCA 输出逐帧归一到单位 RMS（无可学习缩放）后再乘门 | 不归一化时增益转进投影矩阵的少数方向（最大奇异值 0.7 → 120–230），门缩到约 1e-3 抵消 |
| `speaker.train_dropout: 0.5` | 训练时一半 TS 样本关掉 ECAPA（验证与推理照常使用） | 让长注册路径自己承担说话人身份，不只依赖 ECAPA |

开发机 2 卡 8000 步（07-gca-fix2）：GCA 门从第 4000 步起上升（0.01 → 0.023–0.042），把 GCA 的注册换成他人的比关掉 GCA 更差
（TS CE +0.054 对 +0.032），不同注册的记忆平均余弦 0.087。

普通 ASR 和热词 ASR 关掉注册路径（mode NONE）：注册路径照常在混合音开头 3 秒上计算再乘 0，
保证每一步所有可训练参数都有梯度，DDP 不需要 `find_unused_parameters`。注册不进入 LLM 上下文，
所以组批只按待识别音频计时长。

## 数据接入

`LongEnrollCatalogDataset` 继承 `CatalogSwiftDataset`：记录选择、解码、加噪 / 混响 / 变速、热词抽样和
Qwen3-ASR 答案文本都沿用框架，只把输出换成 v5 输入：

- TS-ASR：系统提示换成 v5 的 “The target speaker's enrollment is provided separately. …”，负样本答案为
  `language None<asr_text>`；
- 普通 / 热词 ASR：保留记录自己的系统提示（空 / `Hotwords: …`）；
- mel 用 Qwen3-ASR 特征提取器在 dataloader 进程里算（5 分钟注册约 0.13 秒），以 bf16 传回。

为此对框架做了这些改动：

- `CatalogSwiftDataset(waveforms=True)`：训练也返回波形数组，不经 WAV 字节往返；
- m4a / mp4 / aac / mp3 带起止时间的片段先 seek 再解码（`catalog_dataset._seek_decode`）。原来 audioread 从文件头
  解到目标位置，长文件靠后的 5 分钟片段要 1～6 秒，现在约 0.7 秒；Emilia2 m4a 上与原解码的差 ≥ 44 dB SNR；
- `sampling_cost` 对 `ts_asr` 不计注册槽，含 `ts_asr` 的数据集索引身份加 `ts_cost: mixture`；
- 启动器冻结数据配置时把 `exclude_records` 也转成绝对路径。

验证集用 `validation_samples` 抽样后按时长排序、按 `batching` 上限切批（`CatalogBatchSampler` 只用于训练）。

## 训练

`python -m open_audio_llm.longenroll.train`，由 `open-audio-llm train` 以 torchrun 启动：

- `--deepspeed zero1`（正式配置）：bf16 权重，fp32 主权重与 AdamW 状态由 DeepSpeed ZeRO-1 分片，梯度裁剪与学习率调度
  也由 DeepSpeed 执行；冻结的 ECAPA 保持 fp32。`--deepspeed none`：fp32 权重 + bf16 autocast，torch ZeRO-1；
- bf16 autocast 写在模型 `forward` 里（DeepSpeed engine 会关掉外层 autocast）；默认融合 AdamW（`--adam_fused`）；
- 学习率三组（正式配置）：新增模块 `--learning_rate` 1e-4；门控（同上乘 `--gate_lr_mult`，1e-4）；主干 `--backbone_lr`
  1e-5（编码器第 0–23 层与 LLM，与 v3 全参 SFT 相同）；前 `--warmup_steps`（正式配置 5000 步）线性 warmup，之后余弦退火到 `--min_lr_ratio`（0.1，即新增模块 1e-5、主干 1e-6）。weight decay 0.1（`--weight_decay`），与 v3/v4 的 Hugging Face Trainer 一样不加在 bias 与 norm 权重上（这里按参数维度判断，所有 1 维参数都不加），门控也不加。新增模块共 65.9M：
  注册压缩与查询（43.6M：每窗 32 槽的压缩器、深度归一化、Key/Value 投影，8 个全局与 24 个局部查询及三组交叉注意力块）、
  4 层 GCA（21.0M）、ECAPA 投影 MLP（1.25M），门控是 4 个 GCA 门与 1 个 ECAPA 门（标量）；
- 日志写 `training/logging.jsonl`（`loss`、`grad_norm`、`learning_rate`、`memory(GiB)`、`train_speed(s/it)`；验证
  `eval_loss`、分任务 CE、解码错误率）与 `training/performance-rank0.jsonl`（吞吐、数据等待、前向 / 反向 / 优化器步时间、
  分任务 loss、门控值），CPU 记录器同步到 W&B；
- 验证：分任务 CE（`ts`、`ts_absent`、`asr`、`hotword`），每卡在整个验证集上按间隔抽 `--decode_samples` 条贪心解码，
  中文按字、英文按词算错误率，TS 负样本统计非空输出比例；
- checkpoint：DeepSpeed 模式为 DeepSpeed 布局（bf16 模型 + 各卡 fp32 主权重 / AdamW 分片，约 28 GB）；`none` 模式为
  `model.pt`（可训练参数，fp32）与 `optimizer.rank{r}.pt`。两种都最后写 `trainer_state.pt`（调度器、步数、卡数、Catalog
  采样游标）。正式配置每 2000 步保存、保留 3 个（ACP 任务运行满 24 小时会被挂起，最多损失约半小时）。
  `--memory_probe N` 用最坏批次（30 秒混合音、5 分钟注册、长答案、配方条数上限）跑 N 步报告峰值显存后退出。
  `--resume_from_checkpoint` 续训要求卡数相同；随机数状态不保存，续训与不中断不是逐位一致。`--init_from_checkpoint` 只载入权重
  （任意卡数，两种布局都行），优化器、调度与数据位置重新开始。

## 推理（vLLM）

部署推理分两段，与框架里 Target SOT 的做法相同：PyTorch 前端把请求音频算成音频嵌入，vLLM 用 Qwen3-ASR 的 LLM 从嵌入解码文本。

| 步骤 | 入口 | 环境 |
|---|---|---|
| 导出 | `python -m open_audio_llm.longenroll.export --checkpoint <checkpoint-N> --model <Qwen3-ASR-1.7B> --ecapa_model <ECAPA> --model_config <longenroll_v5.yaml> --output_dir <新目录>` | 训练环境，CPU 约 80 秒 |
| 推理 | JSONL worker `python -m open_audio_llm.longenroll.infer --model_dir <导出目录> --input requests.jsonl --output predictions.jsonl --summary worker.json`；Python 接口为 `integrations/vllm/longenroll.py` 的 `LongEnrollVLLM` | vLLM 环境（vllm 0.18.0、qwen-asr 0.0.6），`PYTHONPATH` 指向本仓库 `src` |
| 评测与一致性核对 | `python -m open_audio_llm.longenroll.evaluate --data_config ... --split validation\|evaluation --samples N --model_dir ... --output_dir ... --worker_python <vLLM 环境 python> --backends vllm torch` | 训练环境（读 Catalog、打分，不加载权重）；worker 在 `--worker_python` 的独立进程里，可见多张卡时每张卡一个 worker、交错分到请求；`--prepared <之前的输出目录>` 复用已准备好的请求与音频 |

- 导出：两种 checkpoint 布局都能读（`model.pt`；DeepSpeed 的 `mp_rank_00_model_states.pt`，冻结部分也取训练时的值）。
  导出目录顶层是 bf16 的 Qwen3-ASR checkpoint，`config.json`、`generation_config.json` 与 tokenizer / processor 文件原样复制自基座；
  `longenroll/` 子目录存注册编码器与其冻结层副本、GCA、门控、ECAPA 投影（bf16）和 ECAPA（fp32）。vLLM 只读顶层权重。
  写完后按前端和 vLLM 的读法读回，逐个张量与 checkpoint 比对；输出目录已存在时拒绝执行。
- 前端（`longenroll/frontend.py`）与训练模型共用条件化模块的构建函数和编码代码：注册在 30 秒窗内做注意力，混合音全局注意力，
  与训练、验证相同（vLLM 自带的 Qwen3-ASR 音频塔按 8 秒窗计算，这里不用它）。ECAPA 输入与验证相同：注册不超过 10 秒整段，
  更长的取开头 10 秒。普通 / 热词 ASR 关掉注册路径。mel、ECAPA fbank 与 prompt 由 `longenroll/features.py` 构造，训练数据也用它。
- vLLM：设置 `OPEN_AUDIO_LLM_ENABLE_QWEN3_ASR_EMBEDS=1`，引擎进程通过 entry point 加载本仓库插件，选用接收 `audio_embeds` 的
  Qwen3-ASR 类。prompt 是训练时的 token id，只含一个 `<|audio_pad|>`，由 vLLM 按嵌入长度展开；贪心解码，遇到 151643 / 151645 停止。
  请求过长（prompt + `max_new_tokens` 超过 `max_model_len`）时报错，不截断音频。
- 请求每行 `{"id", "task": "ts"|"asr"|"hotword", "mixture": 音频, "enrollment": 音频（仅 ts）, "hotwords": "a,b"（仅 hotword）}`，
  音频是 soundfile 能读的路径或 `{"path", "start", "duration"}`，读入后转单声道 16 kHz。输出每行 `{"id", "backend", "language", "text",
  "raw", "token_ids", "finish_reason"}`；摘要记后端、版本、解码参数和各段耗时。读音频和算 mel / fbank 用 `--feature_workers`
  个线程（默认 4；串行时占 worker 总耗时 62%），前端每次处理 `--frontend_batch` 条（默认 16，按混合音长度分组）。
- `--backend torch` 是参考实现：从同一导出目录构建训练模型 `LongEnrollTSASR`，用 Transformers `generate` 逐条贪心解码。
  两个后端的前端分组相同，嵌入完全一致，差别只在 LLM 解码。

## 开发机实测

`runs/longenroll-v5-dev-20261016`（本地，2 张 A800，每卡每批约 12 条，ZeRO-1）：

- 400 步：loss 1.50 → 0.27；验证 loss 2.09 → 0.73，TS 负样本 CE 8.17 → 0.19；峰值显存 31–36 GiB/卡；
- 速度：数据等待 < 1 毫秒，每条样本 worker 0.13 秒。LLM 每卡每秒 token（含 padding）：fp32 + 非融合 AdamW 约 2640，
  融合后约 3425，bf16 + DeepSpeed、每批 14 条约 4530（AmphionASR v3 continue_sft_turn_taking_police 为 3765）；
- 显存：bf16 + DeepSpeed 每批 14 条实测峰值 25 GiB，最坏批次 30.5 GiB；fp32 + autocast 每批 14 条最坏批次 > 39.5 GiB；
- 启动时为全量配方建采样计划约 8 分钟。

推理（`tasks/03-export`、`tasks/04-vllm`，checkpoint-400，4 号卡，vLLM 显存占比 0.35）：500 条验证样本（混合音 6582 秒、
注册音 23498 秒）vLLM worker 14.5 秒，每秒 34 条，RTF 0.002；其中前端 8.8 秒、特征（4 线程）2.5 秒、vLLM 解码 3.0 秒。
训练实现（Transformers `generate` 逐条）346 秒。两者的音频嵌入逐位相同，错误率一致（ASR 6.51% / 6.59%）；
逐 token 相同 87%；在第一个决策（语言 token）就分开的 16 条，torch 前两名的差都 ≤ 0.25（bf16 在该量级两个刻度以内），
vLLM 自身两次执行之间也只有 94% 相同。

8 卡一个 epoch 490258 步（配方每批 14 条；12 条时为 571131 步），按每步 0.8～1 秒估计约 4.5～5.7 天。

## 限制

- vLLM 推理目前是 JSONL worker 与 Python 接口，还没有 HTTP 服务，也没有接入 AmphionEval（本机未安装 AmphionEval，
  未核对它的 TS-ASR 请求能否单独传注册音频）。`evaluate.py` 在 Catalog 的验证 / 测试划分上评测，打分与训练内验证相同，另报漏识率（目标在场、输出为空）；
  整体与按来源（数据配置里的 `label`，没有时为 dataset@version:split）各出一份。
- v5 验证集没有热词任务，vLLM 一致性核对只覆盖 TS 与普通 ASR；热词请求与普通 ASR 只差系统提示，其构造由单元测试覆盖。
- LLM 默认开梯度检查点；显存充足时关掉可省反向里的一次重算。
