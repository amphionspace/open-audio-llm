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
