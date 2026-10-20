# 07-batch24-acp：每卡 batch 24 × 梯度累积 1，降低主机侧开销

- 起因：06-packed-acp 显示 Liger + 打包让每个 microbatch 的 GPU 计算少了 35%，但每步耗时没变，瓶颈在主机侧（Python 下发 kernel 的开销）。每个 microbatch 的主机开销大致固定，所以把 microbatch 数减半。
- 输入：`configs/batch24-acp-1500.yaml`（由 06 的配置派生，Liger + 打包），数据配置 `configs/train-data-v7set4c-b24.yaml`（029 配方；每批 24 条、填充后时长上限 960 s、累积 1）。全局 batch 仍为 192。跑 1500 步，不存 checkpoint。
- 批次计划离线核对（`plan-check/`，真实采样器，96000 条样本）：
  - 更新次数 505 → 506；每次更新的样本数 190.2 → 189.7；装满的批次占比 96.1% → 95.7%；
  - 更新内任务组成偏差 TV 0.315 → 0.316，没有被单一任务占满的更新。
  - 批次的划分方式不同，所以与 029 不是同批次对照。
- 完成条件：1500 步跑完；取预热后的第 1001–1500 步，与 029 同区间对比 s/步、每个 microbatch 的主机侧与 GPU 时间、显存峰值；loss 量级正常；W&B 远端核验通过。
- 注意：`max_steps=1500` 会让余弦调度在第 500 步后比 029 衰减得更快，只影响 loss 对照，不影响测速。
- 进展：2026-10-19 提交，job `pt-50zzicsy`；13:18Z 开始训练，14:04Z 1500 步跑完，执行状态 completed，W&B 远端 finished（`attempts/001/wandb-verification.json`）。

## 结果（证据：`attempts/001/compare-029-by-phase.json`）

| 阶段 | 029 每步 | 07 每步 | 07 每步 GPU 时间 | 07 每步等数据（最慢 rank） |
|---|---|---|---|---|
| 第 1–100 步 | 7.01 s（共 12 分钟） | 7.05 s（共 12 分钟） | 1.67 s | 38 s（冷启动） |
| 第 101–1000 步 | 1.31 s | 1.30 s | 0.68 s | 0.55 s |
| 第 1001–1500 步 | 1.17 s | 1.07 s | 0.68 s（029：1.08 s） | 0.33 s（029：0.03 s） |

- 稳态只快 9%。GPU 时间少了 37%，但空出来的时间变成了等数据。显存峰值 45.3 GB。
- 原假设“每个 microbatch 有固定的主机侧开销”不成立：每个 microbatch 的主机侧时间随样本数翻倍（0.71 → 1.38 s）。
- 运行中用 py-spy 同时采样 8 个 rank（`/workspace/workspace/mingdong/.local-tools/py-spy/out-b24/`）：
  - rank 主线程的“主机侧时间”大部分是同步等待，不是下发 kernel：DeepSpeed 吞吐计时器在 forward 开头 `torch.cuda.synchronize()`，
    而它前面有 transformers 每步一次的跨卡 `gather`（统计监督 token 数），所以要等最慢的 rank 取到数据。
  - 同一 40 s 窗口：rank 699 有 25% 时间卡在取数据，其余 7 个 rank 各有约 28% 时间在 forward 开头空等。另一个窗口这一比例是 56%。
  - 主进程 CPU 接近 100% 是 CUDA 同步的忙等，此前据此判断“主机侧瓶颈”是误判。
  - 慢的数据 worker 57% 的时间在 `catalog_resolver._resolved_directory` 列目录，另一个 worker 34% 在 `sample_hotwords`。
- 结论：029 稳态是 GPU 瓶颈（每步 ≈ GPU 时间）；Liger + 打包把 GPU 时间降到 0.68 s 后，瓶颈换成了数据加载。batch 24 本身不带来收益。
- 后续：08-data-fix-acp。
