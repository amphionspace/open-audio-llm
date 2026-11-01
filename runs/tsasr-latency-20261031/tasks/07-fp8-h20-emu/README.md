# 07 FP8 量化损失（H20 数值，A800 上模拟）

- 问题：线上 TS-ASR 部署在 H20 上；如果打开 vLLM 在线 FP8（`--quantization fp8`），准确率损失多少。
- 为什么不用 02 的 FP8 结果：02 在 A800 上走的是 Marlin，只量化权重，激活仍是 bf16。H20（sm90）走的是 W8A8，激活也量化：
  - 权重逐张量 FP8；
  - 激活逐 token 动态 FP8，因为 `cutlass_fp8_supported()` 为真；
  - `CutlassFP8ScaledMMLinearKernel` 做 FP8 乘、FP32 累加。
  - 依据：vLLM 0.18 `fp8.py`、`kernels/linear`；本机 `ops.cutlass_scaled_mm_supports_fp8(90)=True`，sm80 为 False。
- 方法：`shared/fp8-emu/fp8_sm90_emu.py` 在每个 vLLM 进程里打补丁，经 `sitecustomize.py` 注入，服务配置置 `OAL_FP8_SM90_EMULATION=1` 才生效。
  - 权重和激活仍由 vLLM 自己的代码按 sm90 的方式量化，只把 GEMM 换成"FP8 值放在 bf16 里做乘法、FP32 输出、乘回缩放系数"。
  - FP8 值在 bf16 中精确表示，所以与 H20 的差别只剩累加顺序和 FP8 张量核心累加器的位宽。
- 模拟精度（`scripts/check_fp8_emu.py`，结果在 `results/gemm_check.json`）：用 1.7B 模型尺寸的一个投影层（2048→6144）测。
  - 模拟结果相对精确结果的误差为 1.9e-5，只有 0.006% 的元素差 1 个 bf16 最小单位。
  - FP8 本身相对 bf16 的误差为 3.4%，约是模拟误差的 1800 倍。
- 评测范围：10.28 版本交付包验收用过的全部全量集，每项都与当时的 bf16 run 用同一配置（`configs/fp8emu/`）：
  - TS v12（26,750 条）；
  - 普通 ASR 全量（253,415 条）；
  - 热词全量（47,116 条）；
  - 警务四集（16,358 条）、警务术语单说（3,264 条）；
  - babble v3（3,200 条）、CHiME-4 et05_real（1,320 条）。
- 比较方法：
  - 两边都从 `predictions.jsonl` 按 AmphionEval 评分协议 v2 重新打分（`scripts/compare_fp8.py`），并给出配对 bootstrap 95% 区间。
  - v12 用 `scripts/compare_gate.py`，与 02 同口径，含在场判断翻转数，噪声参照是同权重重跑的 50 条。
  - 警务用 amphiondata 质检规则（`/workspace/workspace/mingdong/scratch/ts033/police_metrics.py`）。
- 入口：`scripts/run_fp8emu.sh`（v12、警务、babble、CHiME-4）、`scripts/run_fp8emu_rest.sh`（热词、ASR 全量两分片）、`scripts/run_bf16rerun.sh`（噪声参照）。日志在 `logs/`，执行状态在 `logs/status.txt`。
- 完成条件：七项全量结果齐全，各自与 bf16 配对比较；W&B 远端核验通过。
- 限制：
  - 不测速度，速度只能在 H20 上测。
  - 不含 FP8 KV cache（`--kv-cache-dtype fp8`）。在 H20 上它走 FlashAttention-3 FP8，Q 也会量化，这里没有模拟。
  - H20 的累加顺序和 FP8 张量核心累加器位宽没有模拟。
  - 服务设置与基线不同（共享卡显存紧）：`--max-num-batched-tokens 4096`、`--max-num-seqs` 64/128、显存比例 0.14；ASR 全量和热词客户端 `batch_size` 64（基线 32）。下面的 bf16 重跑用同样的服务设置，它与基线的差别属于抖动量级。
  - ASR 全量少 1 条：`gigaspeech:test:YOU1000000114_S0000006` 有 4266 token，超过 4096 编码器缓存被服务端拒绝，所以分片 2 的 run 状态是 failed。这条参考文本为空，不计入 CER，对误输出统计最多差 1 条。
  - ASR 全量没有独立噪声参照：10-28 那个 failed run 与最终 run 的 205,488 条预测和耗时完全相同，是续跑复用，不是重跑。没有再跑一遍，因为 FP8 差值的 95% 区间已包含 0。

## 结果

bf16 → FP8（H20 数值），错误率单位为百分点。噪声是同一权重的 bf16 重跑相对基线的差异。

| 评测集 | 条数 | bf16 → FP8 | 差值 95% 区间 | 预测有变化 | 噪声：差值 / 变化 |
|---|---|---|---|---|---|
| TS v12 CER | 26,750 | 13.07 → 13.08 | [−0.096, +0.124] | 6.4% | −0.05 / 2.3% |
| 警务四集 | 16,358 | 0.255 → 0.262 | [−0.003, +0.016]；对重跑 [+0.002, +0.019] | 5.4% | −0.004 / 1.0% |
| 警务术语单说 | 3,264 | 0.313 → 0.337 | [0, +0.064]；对重跑 [−0.016, +0.057] | 0.2% | +0.008 / 0.03% |
| babble v3 | 3,200 | 2.111 → 2.192 | [0, +0.161]；对重跑 [+0.004, +0.162] | 2.6% | −0.004 / 0.5% |
| CHiME-4 et05_real | 1,320 | 7.166 → 6.944 | [−0.540, +0.057]；对重跑 [−0.548, +0.241] | 13.6% | −0.062 / 4.0% |
| 热词全量 | 47,116 | 4.215 → 4.252 | [+0.009, +0.066] | 6.8% | 见下 |
| 热词 8,000 条子集 | 8,000 | 2.987 → 3.002 | [−0.038, +0.069] | 5.8% | +0.021 / 3.8% |
| 普通 ASR 全量 | 253,414 | 4.292 → 4.301 | [−0.004, +0.021] | 6.3% | 无独立重跑 |

其他指标（bf16 / FP8 / bf16 重跑）：

- v12：漏识 2.85 / 2.93 / 2.88；误输出 3.28 / 3.29 / 3.34；在场判断翻转 FP8 99 条（变空 54、变有 45），重跑 50 条，Marlin 权重量化 80 条。
- 警务 v6 扩展集（2,000 条）：整句准确率 98.45 / 98.30 / 98.50，联合槽位都是 100。
- 警务 v6 验收集（3,773 条）：整句和槽位 99.973 / 99.947 / 99.973（FP8 多错 1 条）。
- 术语 v5（683 个）：命中 677 / 674 / 677。增量集（10,143 个）：10124 / 10117 / 10124。术语单说（3,264 个）：3241 / 3239 / 3240。
- babble 误输出 2.17% → 2.33%，来自 crowd-fsd（7.5% → 8.5%，200 条里多 2 条）；重跑不变。
- 热词（全量）：有列表时召回率 aishell 99.84 → 99.88、commonvoice 98.24 → 98.30，干扰词命中 107 → 113。
- 普通 ASR 全量：34 个数据源里 25 个变差、9 个变好，宏平均 4.920 → 4.910。变差最多的是 `police_synthetic_zh_accent:test_cosyvoice3`（+0.121）、`talcs`（+0.087）；变好最多的是 `wenetspeech_chuan:test_hard`（−0.879）。gigaspeech 非语音误输出 33.77% → 33.92%（5,445 条里多 8 条）。

证据：`results/` 下 `v12_compare.json`、`compare_small_with_noise.json`、`compare_part2.json`、`compare_hot_subset.json`、`compare_asrfull.json`、`police_metrics.json`、`police_metrics_rerun.json`。

W&B：12 个评测 run 经代理补录，加 1 个汇总 run [`fp8-h20-emu-comparison-20261101`](https://wandb.ai/1016097967-amphion/open-audio-llm/runs/fp8-h20-emu-comparison-20261101)。远端状态和指标已核对（`results/wandb_verify.json`）；分片 2 因 1 条被拒状态为 failed，只有覆盖率，ASR 全量合并结果在汇总 run 里。

## 进展

- 2026-11-01：模拟与核对完成。
  - 第一次启动时 GPU 5/4 只剩 8/4.75 GiB，vLLM 拒绝启动。模拟补丁已在 API 和引擎两个进程里装上。
  - 改为把量化后的 FP8 权重值直接以 bf16 存放，替换原 FP8 张量（数值不变，显存与 bf16 模型相同）。
  - 换到 GPU 3/2，`gpu-memory-utilization` 设 0.12，重新启动。
  - 第二次启动时 Inductor 把激活量化编成 Triton 内核，但 sm80 上的 Triton 不支持 E4M3（`fp8e4nv`）。
    - 改为把激活量化和 GEMM 合并成一个 torch.compile 看不进去的独立算子。
    - 量化部分逐行照搬 `QuantFP8.forward_native`。这正是 sm90 默认运行的公式：`custom_ops` 为 `["none"]`，FP8 融合 pass 都是关闭的。
  - 单元测试：结果与"原生量化 + 精确乘法 + 一次 bf16 舍入"相比，只有 0.009% 的元素差 1 个最小单位；`torch.compile` 编译后与直接执行完全一致。
  - 服务日志确认了三点：
    - 选中的是 `CutlassFP8ScaledMMLinearKernel`，激活缩放为逐 token（`GroupShape(row=1, col=-1)`）；
    - 量化的是 LLM 的 112 个线性层（28 层 × qkv/o/gate_up/down），音频编码器不量化；
    - 模型显存为 3.87 GiB。
- 2026-11-01：全部跑完。
  - ASR 全量和热词在单服务上太慢，第二段改成三路并行（`run_fp8emu_rest.sh`）。
  - bf16 重跑第一次启动被自己的检查拦下：`ready` 要求日志有 FP8 模拟标记，bf16 服务不会有，重试 6 次后退出，白等约 1 小时。修复后 10:33Z 重启，10:54Z 跑完。
