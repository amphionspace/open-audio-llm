# 08-data-fix-acp：修数据加载，让数据跟上 Liger + 打包后的 GPU 步

- 起因：07 显示稳态下 GPU 每步只要 0.68 s，但最慢的 rank 每步要等数据 0.33 s；前 1000 步还在冷启动，第 1–100 步就用了 12 分钟。
- 定位（本地，冷启动的单个数据 worker，07 的数据配置，同一随机种子取 150 条样本）：
  - 共 137.4 s，其中列目录 80.8 s：读一条音频前要列出它所在的整个目录（为了找出其中的符号链接）。
    Common Voice clips 目录有 1,176,080 个文件，列一次 40.6 s；AISHELLMix 两个 326,100 文件的目录各约 11 s；LibriMix 10–14 万文件的目录各 3–4 s。
    每个 worker 进程各自缓存，所以每节点 96 个 worker 每个都要把这些目录列一遍。
  - 热词采样每条 19.4 ms：16 万条的热词池里，每个元素都重新构造一次排除集合。
- 修复（分支 `perf/data-worker-cold-start`，PR #70，2026-10-19 14:40Z 合入 main `5a919dc`）：
  - `catalog_resolver._resolve_one`：按样本读取时只对该文件做一次 lstat，目录本身的解析按目录缓存；整目录枚举（噪声池等）仍用一次列目录。结果与 `Path.resolve()` 相同（已有测试覆盖文件和目录符号链接）。
  - `hotwords.sample_hotwords`：排除集合只建一次。输出和随机数消耗与原来逐条相同（200 组随机输入核对）。
  - 本地复测同样 150 条：137.4 s → 44.0 s，列目录 0 s，热词 6.6 ms/条。
- 输入：`configs/datafix-acp-1500.yaml`。与 06（`packed-acp-300.yaml`，029 配方、同批次序列）相比只改：代码快照 `shared/code-data-fix`（= code-liger-packed + 上述两个文件）、每卡数据 worker 12 → 20、跑 1500 步。
  - worker 数的依据：稳态每条样本准备约 0.38 s，其中 CPU 只占约 25%，其余在等 I/O；每卡每步 24 条，要跟上约 0.7 s 的 GPU 步需要约 13 个满负荷 worker，留 1.5 倍余量应对长尾。节点 CPU 空闲约 57%、内存实际只用了约 111 GB。
  - 每条样本的随机数由样本自身的种子决定，批次由主进程的采样器划分，所以 worker 数不改变训练数据。
- 完成条件：1500 步跑完、W&B 远端核验通过；与 029 分阶段（1–100、101–1000、1001–1500）比每步耗时、最慢 rank 的等数据时间、GPU 时间；loss 与 029 同批次可比（第 500 步后学习率调度不同）。
- 进展：
  - attempt 001（`pt-t6xzgu88`，`configs/datafix-acp-1500.yaml`）：2026-10-19 14:2xZ 提交，未开始训练；15:16Z 手动停掉，换成带 CUDA 时间线的 attempt 002。执行记录仍是 submitted，没有任何产物。
  - attempt 002（`pt-lqup68j4`，`configs/datafix-profile-acp-1500.yaml`）：同上配置，另在第 1201–1203 步记录每个 rank 的 CUDA 时间线（PR #71 的 `--torch_profile_steps`），作为 10-ds-sync-overlap-acp 的对照。2026-10-19 15:2xZ 提交；15:38Z 开始，16:02Z 跑完 1500 步，执行状态 completed，W&B 远端 finished（`attempts/002/wandb-verification.json`）。
  - 测速时排除第 1201–1210 步所在的日志区间（采时间线会拖慢这几步）。

## 结果（attempt 002，证据：`attempts/002/compare-029-*.json`、`attempts/002/cuda-trace-rank0.json`）

| 项 | 029 | 08 |
|---|---|---|
| 第 1–100 步 | 11.7 分钟 | 1.7 分钟 |
| 第 1–1000 步合计 | 31.3 分钟 | 12.5 分钟 |
| 第 1001–1500 步每步（不含采时间线的 1201–1210） | 1.176 s | 0.746 s（1.56×） |
| 最慢 rank 每步等数据 | 0.07–0.53 s（前 1000 步） | 0.003 s |
| 每条样本准备时间 | 0.28–0.31 s | 0.10 s |
| 显存峰值 | 41.9 GB | 29.9 GB |

- 与 029 同批次（`same_batches: true`）。第 1–500 步 loss 平均相对差 0.014%，最大 0.69%。第 1001–1500 步 08 低 1.4%，是 `max_steps=1500` 让学习率提前衰减，属于预期。
- 数据加载不再拖慢训练，每步耗时约等于 GPU 时间，现在是 GPU 瓶颈。

### 第 1201–1203 步 CUDA 时间线（`scripts/analyze_cuda_trace.py`，8 个 rank）

| 项（每步） | 时间 |
|---|---|
| 窗口 | 0.68 s |
| GPU 有 kernel 运行 | 0.55 s（空闲 19%） |
| 计算 kernel | 0.41 s，其中矩阵乘 0.275 s |
| NCCL | rank 0 为 0.139 s，与计算完全没有重叠 |
| 其中 AllReduce | 每步 11 次，单次中位数 5.1 ms，纯通信约 56 ms；其余 35–60 ms 是等其他 rank 到齐（各 rank 计算量不均） |
| 其中 AllGather（优化器之后同步参数） | 22 ms |
| 文本解码器前向的空闲 | 约 44 ms：小 microbatch 的前向受 CPU 下发速度限制（53 ms 的前向里 GPU 只忙 11–17 ms） |
| 其余空闲（反向、编码器、步与步之间） | 约 85 ms，以 50–200 µs 的小空隙为主（每步约 580 个） |

- 配置里虽然是 `reduce_scatter: true`，但 DeepSpeed 的 `use_multi_rank_bucket_allreduce` 默认为 true：一个桶里的梯度只要分属多个 rank，就对整个桶做 AllReduce。
- 由此排出的后续候选：`overlap_comm`（10 在测）；按目标 rank 做 Reduce（`use_multi_rank_bucket_allreduce: false`，等 10 的结果再定）；采样时在 rank 之间均衡 token 数；降低文本解码器前向的 CPU 下发开销。
