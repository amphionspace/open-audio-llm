# 训练性能日志

Catalog SFT 默认开启 `--performance_logging true`。计数来自真正进入 `training_step` 的 batch，不包含 worker 预取但尚未训练的数据。日志不保存音频内容或转写文本。

每个训练输出版本目录增加：

| 文件 | 内容 |
| --- | --- |
| `performance-rank0.jsonl`、`performance-rank1.jsonl` | 每个 rank 的真实吞吐和阶段耗时，频率跟随 `logging_steps` |
| `gpu.csv` | 每秒采样可见 GPU 的利用率、显存、功耗、SM / 显存时钟和设备 UUID |
| `logging.jsonl` | ms-swift 原有的 loss、梯度范数、学习率、验证指标 |

控制台也会输出 rank 0 的 `Training performance` 行。启动时额外记录 Catalog 元数据加载耗时和训练／验证样本数。GPU 监控使用系统已有的 `nvidia-smi`，训练结束或抛出异常后由训练入口停止监控进程；没有 CUDA 或 `nvidia-smi` 时保留 CPU 和数据统计。

原生 Qwen3-ASR 脚本可以用 `PERFORMANCE_LOGGING=false` 关闭性能日志。直接调用 Catalog SFT 入口时使用 `--performance_logging false`。

## 统计口径

所有计数和计时默认属于**当前 rank、当前日志区间**，不是累计到当前 step 的值。

| 字段 | 含义 |
| --- | --- |
| `timestamp` / `rank` / `step` / `updates` | Unix 时间戳、rank、当前优化器步数、本区间更新次数 |
| `samples` / `microbatches` / `sources` | 实际训练的样本数、微批次数、各数据来源的样本数 |
| `audio_seconds` | 所有音频槽位经过波形增强后的实际音频秒数 |
| `interval_wall_s` | 两次性能日志之间的墙钟时间；如期间执行验证或保存，也会计入 |
| `samples_per_s` / `audio_seconds_per_s` | 当前 rank 的真实样本／音频秒吞吐 |
| `batch_fetch_s` / `batch_fetch_fraction` | Trainer 获取本次梯度累积所需 batch 的时间及占区间比例；包含等待预取、CPU→GPU 传输及 batch token 计数，不是纯磁盘 I/O 时间 |
| `decode_s` / `wave_augment_s` | 本区间已训练样本的解码重采样／波形增强累计耗时 |
| `encode_s` | tokenizer、音频特征提取及 SpecAugment 的累计耗时 |
| `prepare_s` / `prepare_cpu_s` | worker 处理已训练样本的累计墙钟／进程 CPU 时间；多个 worker 并行，不能直接与训练墙钟相加 |
| `prepare_max_s` | 当前区间最慢单条样本的准备耗时 |
| `collate_s` | 已训练 batch 的 collator 累计耗时 |
| `forward_backward_host_s` | 包装 `training_step` 测得的主线程累计耗时，包含前向、反向及内部同步，不含外层 optimizer.step |
| `forward_backward_cuda_span_s` | CUDA event 测得的前向／反向流时间跨度，包含 kernel 间的空隙和通信等待，不能当作 GPU 忙碌时间 |
| `update_wall_s` | step begin 到 step end 累计耗时，包含梯度累积和优化器更新；取 batch 通常在 step begin 之前 |
| `tokens` / `tokens_capacity` | attention mask 中有效位置数／组批后的总位置数，包含音频占位 token |
| `supervised_tokens` | labels 中不为 `-100` 的 token 数 |
| `audio_frames` / `audio_frames_capacity` | 提供 feature attention mask 的模板中，有效音频帧数／组批总帧数 |
| `*_padding_fraction` | padding 占比，用 `1 - 有效数 / 总容量` 计算 |
| `tokens_max_length` / `audio_frames_max_length` | 区间内最大组批序列／音频特征长度 |
| `cuda_allocated_gib` / `cuda_reserved_gib` | 当前实际分配／缓存保留显存，GiB |
| `cuda_peak_allocated_gib` | 当前进程启动以来的峰值实际分配显存，GiB |
| `loss` / `grad_norm` / `learning_rate` | 对齐同一 step 的 ms-swift 训练指标 |

CUDA event 只在日志区间结束时同步，不在每个微批次强制同步。GPU CSV 是设备级采样；如同卡另有任务，利用率和显存会混入其他任务，因此性能对照必须独占 GPU。

比较多卡吞吐时，先对齐各 rank 的 step 区间，用总样本数或总音频秒数除以最慢 rank 的区间墙钟时间。不要把某个 rank 的音频吞吐直接乘 world size，也不要拿 Trainer 按静态 batch 推算的 `train_samples_per_second` 代替真实计数。

## 本次 Qwen3-ASR 配方调整

在双 A800 80GB、相同模型和在线增强下，排除前 10 步启动阶段的 80 步对照结果：

| 设置 | 实际样本/秒 | 音频秒/秒 | GPU 2 / 3 平均利用率 |
| --- | ---: | ---: | ---: |
| 每卡 4，累积 2，开启 checkpointing 和 unused 参数扫描 | 18.91 | 97.73 | 43.6% / 33.2% |
| 每卡 4，累积 2，关闭上述两项 | 31.73 | 164.01 | 43.7% / 60.3% |
| 每卡 8，累积 1，关闭上述两项 | 50.09 | 258.17 | 59.5% / 64.2% |

数据等待占比仅 0.1%–0.3%，没有通过增加 worker、缓存音频或关闭在线增强来换取吞吐。第一组与第二组所有已记录 loss 完全一致。第三组改变了动态组批边界，实际样本数稍有差异，因此同时报告按实际音频秒数归一的吞吐。

[原生配方](../../examples/train/qwen3-asr/README.md)采用第三组设置。通用模型的 checkpointing 和 batch 默认值不因此改变。显存较小的设备可降低 YAML 的 `batching.max_samples`、相应修改 `per_device_train_batch_size`，并按需要开启语言模型 checkpointing；冻结的音频塔保留 `vit_gradient_checkpointing=false`。

## 定位瓶颈

先分清每步的时间是花在 GPU 计算、等数据，还是 GPU 空等上，再决定优化什么。只看某一侧会误判：

- **不要把 rank 主进程 CPU 接近 100% 当成 CPU 瓶颈。** CUDA 同步是在 CPU 上原地空转等待，同样会把 CPU 占满。`forward_backward_host_s` 也包含这些等待。
- **每一步开头有一次跨卡对齐。** transformers 每步会 `gather` 一次监督 token 数，DeepSpeed 吞吐计时器又会在 forward 开头同步，所以最慢 rank 取数据的耗时会落到所有 rank 头上。要看的是每个日志区间里**最慢 rank** 的 `batch_fetch_s`，不是平均值。
- **按训练阶段分开比较。** 数据 worker 的缓存要经过一段时间才能热起来，开头几百步和稳态的耗时差别很大。对照时分开比较第 1–100 步、热身阶段和稳态，不要只看前几百步。
- **单卡计时不能推算整步提速。** GPU 计算省下的时间，可能被等数据、通信或空泡吃掉。

| 现象 | 说明 | 手段 |
| --- | --- | --- |
| 每步耗时明显大于 `forward_backward_cuda_span_s`，最慢 rank 的 `batch_fetch_s` 大 | 等数据 | 用 py-spy `record --idle` 同时采样所有 rank 和数据 worker |
| 每步耗时约等于 `forward_backward_cuda_span_s` | GPU 侧（计算、通信或空泡） | `--torch_profile_steps` 采时间线 |

`--torch_profile_steps 第一步 最后一步` 会在这几次参数更新期间，为每个 rank 写一份只含 CUDA 活动的时间线（训练输出目录下的 `torch-profile-rankN.json.gz`）。不记录 CPU 算子，避免拖慢主机、放大 kernel 间隙。用下面的命令汇总：

```bash
python scripts/analyze_cuda_trace.py <输出目录>/torch-profile-rank0.json.gz --updates <采样的更新次数>
```

它会给出：每步 GPU 忙碌时间（所有 stream 的并集）、NCCL 时间及其中没有与计算重叠的部分、各类 kernel 的耗时，以及最长的空闲间隙和它们前后的 kernel。NCCL kernel 的耗时包含等待其他 rank 到齐的时间。看各 rank 中 NCCL 时间最少的那个，才更接近纯通信时间。

## Qwen3-ASR 推荐的提速设置（2026-10）

ACP 单节点 8×A800，ts-robust v7set4c 配方，每卡 batch 12、梯度累积 2，与 029 使用同一批次序列：

| 项 | 029（原设置） | 新设置 |
| --- | ---: | ---: |
| 第 1–100 步 | 11.7 分钟 | 1.7 分钟 |
| 第 1–1000 步合计 | 31.3 分钟 | 12.5 分钟 |
| 第 1001–1500 步每步 | 1.176 s | 0.746 s（1.56×） |
| 显存峰值 | 41.9 GB | 29.9 GB |

第 1–500 步 loss 与 029 的平均相对差为 0.014%，最大 0.69%。

新设置（在已有的 `--audio_encoder_batching true`、`--supervised_head_loss true`、关闭 gradient checkpointing 基础上）：

- `--use_liger_kernel true`：文本解码器换成 Liger 的 RMSNorm、SwiGLU、RoPE kernel（#69）。
- `--packed_text true`：每个 microbatch 拼成一行，不再填充，文本解码器走 FlashAttention-2 varlen（#69）。需要 flash-attn 2.8.3。
- `--dataloader_num_workers 20`：稳态下每条样本准备约 0.1 s，主要在等 I/O，20 个 worker 留有余量。worker 数不改变训练数据。
- 代码层面已默认生效：按样本读音频时不再列整个目录（#70），音频编码器和监督头不再强制 CPU 等 GPU（#72）。

`--use_liger_kernel` 和 `--packed_text` 会改变 BF16 数值舍入。新实验只和同样开着这两项的实验比较，在开始一个新系列时一起打开。

剩余可优化空间（第 1201–1203 步的 CUDA 时间线）：

- 每步 GPU 空闲约 19%。
- 梯度 AllReduce 与计算完全串行，纯通信约 56 ms/步，另有 35–60 ms 是等其他 rank 到齐。
- 小 microbatch 的文本解码器前向受 CPU 下发速度限制，约 44 ms/步。

证据和后续对照记录在 `runs/train-speed-20261019`（08、10）。
