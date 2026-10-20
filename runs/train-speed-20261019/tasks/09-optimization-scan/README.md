# 09-optimization-scan：#69/#70 之后还剩哪些提速空间（只读扫描，未改代码）

- 输入：029（ts-robust v7set4c，35000 步，11.2 小时）、07（Liger + 打包 + b24）的性能日志、`gpu.csv`、`storage-uploads.json`；
  07 运行中 8 个 rank 的 py-spy（`/workspace/workspace/mingdong/.local-tools/py-spy/out-b24/`）；本地数据 worker 与数据预检的 py-spy。
- 完成条件：列出候选项，每项附证据和估算收益；需要节点才能定量的项单独标出。
- 进展：2026-10-19 完成扫描。

## 029 的时间账

| 项 | 时间 | 说明 |
|---|---|---|
| 稳态训练 | 约 10.4 小时 | 每步约 1.08 s，等于 GPU 时间，是 GPU 瓶颈 |
| 启动到第 1 步 | 4.4 分钟 | 数据预检 2.2 分钟 + 加载模型、构建数据集 2.2 分钟 |
| 冷启动多花的时间（前 1000 步） | 约 14 分钟 | 主要是逐 worker 列目录，#70 已修 |
| 每 1000 步存 checkpoint | 每次约 20 秒，共约 12 分钟 | 上传在后台线程，不阻塞 |
| 结束收尾 | 10.4 分钟 | 其中约 9 分钟同步上传最终 checkpoint 的续训状态 |
| Python GC 停顿 | 每 rank 共约 20 秒 | 可忽略 |

## 数据侧：#70 之后不再是稳态瓶颈

- 029 在 ACP 上每条样本的准备时间随训练下降：第 1001–1500 步 0.228 s，第 5001–6000 步 0.132 s，第 34001–35000 步 0.118 s。
  主要原因是逐 worker 的目录列表缓存要很久才能填满，#70 已经去掉。
- 本地用真实采样器（rank 0，第 2000–2029 个批次，355 条）测得每条 0.146 s，其中数据增强 0.077 s、解码 0.032 s。
  证据：`evidence/worker-real-sampler-*.{json,txt}`。
- Liger + 打包后 GPU 每步约 0.68 s。b12×GA2 下每卡每步 24 条、12 个 worker，每条预算约 0.34 s。稳态 0.12–0.15 s，约有 2–3 倍余量。
- 更正：此前说“aishellmix 每条 1.9 s”是测量假象。那一轮的第一条样本恰好是 aishellmix，承担了每个进程只发生一次的模块导入。
  训练中主进程已导入 `qwen_asr`，数据 worker 由 fork 继承，不会付出这部分。

## GPU 侧：剩下的主要空间

按 checkpoint 配置与 07 实测（每卡每步 24 条、6612 个文本 token，其中约 4500 个是音频 token）估算：

| 部分 | 每步计算量 | 估算耗时 |
|---|---|---|
| 文本解码器（1.41B 非嵌入参数） | 约 5.7e13 FLOPs | 约 0.34 s（单卡计时推算） |
| 音频编码器（24 层、d=1024）+ 冻结卷积前端 | 约 1.15e13 FLOPs | 约 0.1 s |
| 梯度 reduce-scatter + 参数 all-gather（约 2B 可训练参数） | — | 约 0.06 s，目前与计算串行 |
| 空泡（GPU 上没有 kernel 在跑） | — | 029 利用率 92.3%，约 85 ms/步；GPU 计算缩短后占比升到约 13% |
| 合计 | — | 实测 0.68 s，相当于 A800 bf16 峰值的 32% |

候选项（收益都是估算，需要一次 ACP 上的 GPU 时间线来定量）：

| # | 项 | 证据 | 估算收益 | 改动与风险 |
|---|---|---|---|---|
| G1 | 关掉 DeepSpeed 吞吐计时器的同步：`"timers": {"throughput": {"enabled": false}}` | 每个 microbatch 开头 `torch.cuda.synchronize()`（`engine.py:2986 → timer.py:238`），07 的 rank 主线程有 27% 时间停在这里；DeepSpeed 文档说明它会降低性能；计时结果没有被使用 | 单独开小，与 G3 合起来能消掉大部分空泡（最多约 13%） | 只改 DS 配置，数值不变 |
| G2 | `overlap_comm: true`，让梯度归约与 backward 并行 | 现在归约在 backward 结束后串行执行（`allreduce_gradients → reduce_gradients:938`，即 `overlap_comm=false` 分支） | 约 0.03 s/步（约 4–5%） | 只改 DS 配置；多约 2 GB 显存 |
| G3 | 去掉前向里的隐式同步 | `retention.py:55/58/65` 的布尔索引和 `nonzero`；`audio_batching.py:64/114` 的 `torch.tensor(list, device=cuda)`（阻塞拷贝）与 `[valid]` 布尔索引；`sep_token.py` 的 `.tolist()`；transformers `_prepare_inputs` 的阻塞 H2D | 与 G1 合计最多约 13% | 改代码；位置分散，需要时间线确认哪些在关键路径上 |
| G4 | 音频编码器改用 FA2 varlen（打包，不再填充） | 编码器在填充后的 batch 上跑带 mask 的 SDPA；音频帧填充约 18–22%；flash-attn 已装好 | 约 3–5% | 改代码；需核对与现有实现的数值一致性 |

已排除：
- batch 24 × GA 1：GPU 时间只少 3%（07：0.676 vs 0.70 s/步），显存多 15 GB。
- rank 0 的梯度范数循环：8 个 rank 在这一阶段的占比都是 22–26%，实际是在等 backward 和归约完成，不是 rank 0 独有的开销。
- GC 停顿、日志、W&B、nvidia-smi 采样：都可以忽略。

## 每次启动和结束的固定开销

| # | 项 | 证据 | 收益 | 改动 |
|---|---|---|---|---|
| F1 | 最终 checkpoint 的续训状态改到 GPU 任务结束后再上传 | 029 结尾 9 分钟 8 卡空闲（`checkpoint_upload.py` 的 `on_train_end` 同步上传后删除本地副本） | 每次训练省约 9 分钟节点时间 | 改流程：由 CPU 侧从共享卷 /workspace 上传 |
| F2 | 数据预检按数据配置的内容缓存结果 | 每次启动都要单进程跑 2–3 分钟，期间 8 卡空闲；续训时数据配置不变 | 每次启动省 2–3 分钟 | 改代码 |
| F3 | 数据目录声明（catalog）加载缓存 | `load_catalog` 38 秒（`read_declarations` 33 秒），预检和每个 rank 各读一遍 | 每次启动约 1 分钟 | 改 audio-data-contract 或加缓存 |
| F4 | 预检里避免为 `validate_language` 导入整个 `qwen_asr` | 预检进程首条样本导入耗时 25 秒 | 每次启动约 25 秒 | 小改，但要复制上游的语言表 |

## 建议

下次有节点时跑一个短任务：029 配方 + Liger + 打包 + #70，在第 1000 步之后用 `torch.profiler` 采几步 GPU 时间线，同时对 G1、G2 各做一组对照，定量空泡、通信和编码器的占比，再决定 G3、G4 是否值得改。
固定开销里，F1 的收益最确定、与训练数值无关。

## 后续进展（2026-10-19）

| 项 | 处理 | 结果 |
|---|---|---|
| 数据冷启动（列目录、热词） | #70 | 08：第 1–1000 步 31.3 → 12.5 分钟 |
| GPU 时间线采集 | #71 `--torch_profile_steps`，`scripts/analyze_cuda_trace.py`（#76） | 08、10 各采 3 步 |
| G3 前向隐式同步 | #72 | 编码器 11 → 0 次，监督头 3 → 1 次；尚未单独测墙钟收益 |
| G1 计时器同步 + G2 `overlap_comm` | 10 | 稳态快 1.1%（中位数 0.4%），不推广 |
| 结束时 rank 进程约 107 s 才退出 | #79 | 原因是 torch 为常驻 worker 注册的 atexit join 逐个等满 5 s；本地 4 个 worker 的退出时间 22.5 s → 2.4 s |
| F1 最终续训状态上传移出 GPU 任务 | 未做 | 需要一个 GPU 任务之外的上传执行者；每次长训约省 9 分钟 |
| F2 预检缓存 | 不做 | 预检是数据正确性门槛；按配置缓存会漏掉磁盘上新坏掉的数据 |
| 代码组织 | #73 删除无引用模块（约 1,030 行）；#74 vLLM 插件同名类重复注册时报错；#75 奖励插件不再改 `sys.path`；#77 取消跟踪快照里的 83 个构建产物；#78 HF 编码器测试移入 CI；#80 README 支持矩阵与代码一致 | — |

### 时间线显示的剩余空间（08 / 10，每步约 0.68 s）

按阶段拆分 08 第 1201–1203 步 rank 0 的 GPU 空闲（`scripts/split_trace_phases.py`，证据 `evidence/trace-phases-08-10.txt`，阶段边界按标志 kernel 近似）：

| 阶段 | 跨度 ms/步 | GPU 空闲 ms/步 | 50–200 µs 小空隙 |
|---|---|---|---|
| 音频编码器前向 | 65 | 17 | 53 个 |
| 文本解码器前向 | 122 | 35 | 272 个 |
| 文本解码器反向 | 222 | 36 | 182 个 |
| 音频编码器反向 | 89 | 24 | 61 个 |
| 梯度归约、优化器、参数 AllGather | 181 | 16 | 11 个 |

- 各 rank 不同步：AllReduce 里等其他 rank 到齐，约 35–60 ms。采样器已经按"条数 × 最长时长"在 rank 之间均衡；更贴近实际 token 数的排序键此前评估过，会让约 20% 的更新只剩单一任务，没有采用。
- 前向和反向里的 GPU 空闲合计约 113 ms/步，是 CPU 逐个下发 kernel 跟不上（前向约 53 ms，反向约 60 ms）。
- 更正：此前把"每步约 580 个小空隙、约 50 ms"归因于 DeepSpeed 逐参数 hook，这是错的。
  - 一半以上的小空隙在前向里，而 ZeRO-1 前向没有 DeepSpeed hook。
  - 07 的 8 个 rank 的 py-spy 里，DeepSpeed 的 `grad_handling_hook` 只占训练循环时间的 0.3–0.7%，约 2–5 ms/步（证据 `evidence/pyspy-07-ds-hook-share.txt`）。
  - 反向里 Liger、FlashAttention 的 Python backward 约占 2.1–2.4%。
- 因此换 torch DDP 不值得做（撤回此前"约快 10–15%"的估计）：
  - 能省的只有 DeepSpeed hook 的约 2–5 ms/步，以及通信与 backward 重叠；重叠在 10 里已验证不改变墙钟。
  - 多出的开销：ZeRO-1 现在每卡只更新 1/8 的参数（优化器 3.9 ms + AllGather 22 ms），DDP 每卡更新全部参数（约 31 ms）；fp32 主权重用 autocast 时，每个 microbatch 要把约 2B 参数转成 bf16（约 12 GB 显存读写），还会增加 kernel 数，而前向本来就受下发限制；显存约多 20 GB。
  - 估计净收益在 ±3% 以内，且会改变数值，需要开新系列。

### 已知不稳定测试

`tests/test_catalog_training.py` 里比较 WAV 字节完全相等的两个测试，在本地全量运行时约 5% 会出现某个样本差 1 个最低有效位，CI 上未出现。已排除：numpy float32 求和受内存对齐影响（实测与对齐无关）、数据管线多线程（没有）。原因未找到。
