# 10-ds-sync-overlap-acp：去掉 DeepSpeed 计时同步，并让梯度归约与 backward 重叠

- 起因（09-optimization-scan）：Liger + 打包后每步 GPU 时间约 0.68 s。其中：
  - 029 稳态 GPU 利用率 92.3%，约 85 ms/步 GPU 上没有 kernel；
  - ZeRO-1 的梯度归约与参数 all-gather 约 60 ms/步，现在在 backward 之后串行执行。
- 输入：`configs/ds-overlap-notimer-acp-1500.yaml`。与 08 attempt 002（`configs/datafix-profile-acp-1500.yaml`）只差 DeepSpeed 配置 `configs/zero1-bf16-no-overflow-check-overlap-notimer.json`：
  - `timers.throughput.enabled: false`：DeepSpeed 吞吐计时器默认在每个 microbatch 的 forward 开头和每步结束各做一次 `torch.cuda.synchronize()`（`engine.py:2986`、`timer.py:238/256`）；它的计时只在 `steps_per_print` 时打印，我们没有使用。
  - `zero_optimization.overlap_comm: true`：最后一个 microbatch 的 backward hook 里分桶做 reduce-scatter，与剩余 backward 计算重叠（`stage_1_and_2.py:1856`）；约多用 2 GB 显存。
  - 两项都不改变计算结果（同样的归约，只是时机不同）。
- 完成条件：1500 步跑完、W&B 远端核验；与 08 attempt 002 比较第 1001–1500 步（排除采时间线的区间）每步耗时、GPU 时间、显存峰值；loss 与 08 逐步一致（同批次、同种子）；用两边第 1201–1203 步的 CUDA 时间线拆分空泡与通信的变化。
- 进展：2026-10-19 15:2xZ 提交，job `pt-rls1n590`；16:03Z 开始，16:27Z 跑完，执行状态 completed，W&B 远端 finished。

## 结果（证据：`attempts/001/compare-08-1001-1500.json`、`attempts/001/cuda-trace-rank0.json`）

| 项 | 08 attempt 002 | 10 |
|---|---|---|
| 第 1001–1500 步每步（均值 / 中位数） | 0.750 / 0.738 s | 0.741 / 0.735 s（快 1.1% / 0.4%） |
| 时间线：GPU 有 kernel 运行 | 0.551 s | 0.507 s |
| 时间线：计算 kernel | 0.411 s | 0.432 s |
| 时间线：NCCL 总时长 | 0.139 s | 0.179 s |
| 时间线：NCCL 与计算未重叠部分 | 0.139 s | 0.076 s |
| 显存峰值 | 29.9 GB | 30.3 GB |

- 同批次（`same_batches: true`），loss 与 08 的平均相对差 0.014%，最大 0.84%。
- 重叠确实把约 0.06 s 的通信藏到了计算后面，但通信 kernel 与计算抢 SM，计算变慢、通信本身也变长，墙钟时间几乎不变。
  每步耗时由各 rank 不同步造成的等待（AllReduce 里等其他 rank 到齐）和 CPU 下发开销决定，单靠重叠解决不了。
- 结论：收益在噪声范围内，不推广这两项配置。
- 退出阶段（另行定位）：两次任务的 rank 进程都在训练结束后约 107 s 才退出，原因是 torch 为常驻 worker 注册的 atexit join 逐个超时，修复见 PR #79。
