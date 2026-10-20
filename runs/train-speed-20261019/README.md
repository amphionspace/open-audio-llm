# train-speed-20261019：ts-robust 训练再提速

## 结论（2026-10-19）

在 029 配方上打开 `--use_liger_kernel true --packed_text true --dataloader_num_workers 20`，再加上 #70、#72、#79 的代码修复：
稳态每步 1.176 → 0.746 s（1.56×），第 1–1000 步 31.3 → 12.5 分钟，显存峰值 41.9 → 29.9 GB，与 029 同批次 loss 一致（08）。
Liger 和打包会改变 BF16 舍入，在下一个新系列开始时一起打开。推荐设置与定位方法写在 `docs/guides/training-performance.md`。

| 项 | 预估 | 实测 | 决定 |
|---|---|---|---|
| 排序窗口 2000 → 8000 | 填充 24% → 12%，提速 8–15% | 按排序键只省 0.7% 计算；每步任务组成略偏 | 不改 |
| Liger kernel（RMSNorm / RoPE / SwiGLU） | 提速 10–20% | 文本解码器前向+反向 1.41×，推算整步约 1.2×；梯度在 BF16 噪声内 | 待 ACP 端到端核对后用于下一次训练 |
| 排序键接近实际 token 长度 | 填充 24% → 更低 | 可算的排序键只省 9–10% LLM token（每步约 5%），约 20% 的更新步被单一任务占满 | 不单独实施 |
| 打包文本批次（padding-free，FA2 varlen） | — | 文本解码器 1.35×；打包与填充 loss/梯度完全相同；FA2 比现状 SDPA 更接近 FP32 | 已实现 `--packed_text`（PR #69 已合入） |
| Liger + 打包的 ACP 端到端核对（06） | 整步 1.35–1.4×（推算） | 与 029 同批次：loss 对齐、显存 41.9 → 29.9 GB、GPU 计算 −35%，但每步耗时不变 | 数值正确。当时判断为主机侧限制，07 查明是误判：主进程在等最慢 rank 取数据 |
| 每卡 24 × 累积 1（07） | 减少每个 microbatch 的主机开销 | 稳态快 9%；GPU 每步 0.68 s，最慢 rank 每步等数据 0.33 s | 不采用（GPU 时间只少 3%，显存多 15 GB）；转修数据加载 |
| 数据冷启动修复 + 每卡 20 个 worker（08，#70） | — | 稳态 1.56×；第 1–1000 步 31.3 → 12.5 分钟 | 采用 |
| 关掉 DeepSpeed 计时同步 + `overlap_comm`（10） | 4–13% | 快 1.1%（中位数 0.4%）；通信被计算盖住约 60 ms，但计算变慢 | 不推广 |
| 训练结束后 rank 进程约 107 s 才退出（#79） | — | torch 为常驻 worker 注册的退出等待逐个等满 5 s；本地 4 个 worker 22.5 → 2.4 s | 已合入 |
| 剩余空间（09） | — | 每步 0.68 s 中 GPU 空闲约 19%：前向和反向里 CPU 下发跟不上约 113 ms，rank 间互相等待 35–60 ms | 换 torch DDP 净收益在 ±3% 以内，不做；最终续训状态上传移出 GPU 任务（每次长训约省 9 分钟）未做 |

## 背景

ts-robust 029（ACP 1 节点 8 卡）约 1.09 s/步，取数据只占 0.3%；LLM token 填充 23.9%、音频帧填充 13.8%。

## 任务

- [01-sort-window](tasks/01-sort-window/README.md)：真实采样器离线对比窗口 2000 / 4000 / 8000 / 16000。
- [02-liger-check](tasks/02-liger-check/README.md)：真实权重、单卡配对计时与 FP32 梯度参照。
- [03-liger-acp](tasks/03-liger-acp/README.md)：Liger 单独的 ACP 核对（`pt-srtumbsm` 被资源池挂起、未启动，已删除，由 06 取代）。
- [04-length-key](tasks/04-length-key/README.md)：8000 条真实编码长度，模拟不同排序键的填充与更新步组成。
- [05-padding-free](tasks/05-padding-free/README.md)：打包文本批次的计时、FP32 参照与完整链路核对。
- [06-packed-acp](tasks/06-packed-acp/README.md)：Liger + 打包的 ACP 300 步核对（`pt-isk606t5` 完成；attempt 001 因存储路径配置错误在启动时失败）。
- [07-batch24-acp](tasks/07-batch24-acp/README.md)：每卡 24 × 累积 1 的 ACP 1500 步（`pt-50zzicsy`），并用 py-spy 定位到等数据。
- [08-data-fix-acp](tasks/08-data-fix-acp/README.md)：修数据冷启动后的 ACP 1500 步，附第 1201–1203 步 CUDA 时间线（attempt 002 `pt-lqup68j4`；attempt 001 未开始即停掉）。
- [09-optimization-scan](tasks/09-optimization-scan/README.md)：剩余提速空间的扫描、时间线拆分与 DDP 评估。
- [10-ds-sync-overlap-acp](tasks/10-ds-sync-overlap-acp/README.md)：关掉 DeepSpeed 计时同步并打开 `overlap_comm` 的 ACP 1500 步（`pt-rls1n590`）。

## 代码

均已合入 main：
- #69 Liger kernel 与打包文本批次（`--use_liger_kernel`、`--packed_text`）；
- #70 按样本读音频时不再列整个目录，热词采样的排除集合只建一次；
- #71 `--torch_profile_steps` 采集指定更新步的 CUDA 时间线；
- #72 音频编码器和监督头不再让 CPU 等 GPU；
- #76 文档 `docs/guides/training-performance.md` 与 `scripts/analyze_cuda_trace.py`；
- #79 训练结束时主动关闭常驻数据 worker。

#69 中 Liger 部分的设计说明：
- transformers 按 `config.model_type` 查 Liger 支持表，`qwen3_asr` 不在表中，原来的 `--use_liger_kernel` 只打一条 info 日志、什么都不换。新增 `integrations/ms_swift/liger.py`，在训练模型实例上替换 thinker 文本解码器的 RMSNorm（含 q/k norm）和 SwiGLU，并替换 qwen_asr 模块级 RoPE 函数；启动日志打印替换数量，找不到可替换模块时报错。
- RoPE 是模块级函数，会同时影响同进程的 retention teacher，所以与 `retention_teacher` 同时开启时直接报错。
- 未替换交叉熵：`--supervised_head_loss` 已只对约 6% 的有监督位置投影，剩余 CE 很小，且 fused CE 不产出 logits，会丢掉 token_acc。
- 依赖：`oal-train` 环境已装 `liger-kernel==0.8.4`（`--no-deps`，torch 2.10 / triton 3.6 满足要求），`pyproject.toml` 的 `swift` extra 固定该版本（接入用到其内部 patch 函数）。
