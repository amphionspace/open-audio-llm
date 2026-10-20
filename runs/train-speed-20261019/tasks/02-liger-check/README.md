# 02-liger-check：Liger kernel 正确性与文本解码器计时（单卡，不训练）

- 输入：`/workspace/data/ae-runs/ts-robust-026-ckpt25000/checkpoint-25000` 的真实权重；合成输入，形状取 029 实测（每个 microbatch 12 条，填充后长度中位约 360 token、填充 24%、有监督位置 262 个）。代码为分支 `perf/sort-buffer-and-liger-kernel`（liger-kernel 0.8.4）。
- 方法：
  - `scripts/liger_microbench.py`：原生与 Liger 两份文本解码器在同一张卡上逐次交替前向+反向（只投影有监督位置，同 `--supervised_head_loss`），5 次预热、30 次计时取中位数。
  - `scripts/liger_grad_reference.py`：同一输入分别跑 FP32 原生（参照）、BF16 原生、BF16 Liger，比较逐参数梯度相对 FP32 的误差。
- 卡：本机 GPU 4（该卡上是本账号一个空闲 vLLM，0% 利用率，剩约 34 GB；未停止它）。
- 进展：attempt 001 完成（2026-10-19）。首次运行因脚本把共享权重的 `embed_tokens` 梯度也拿来比较而报错（`run-failed-grad-key.log`），修正比较范围后重跑。

## 结果

| 形状（条 × token） | 原生 s | Liger s | 加速 | 每个 microbatch 省 |
|---|---|---|---|---|
| 12 × 360 | 0.324 | 0.228 | 1.42× | 0.096 s |
| 12 × 480 | 0.431 | 0.305 | 1.41× | 0.126 s |
| 12 × 600 | 0.542 | 0.386 | 1.40× | 0.156 s |

- 替换了 28 层的 RMSNorm（含 q/k norm，共 113 个）、28 个 SwiGLU MLP，以及 RoPE。
- loss 与原生一致（14.318 vs 14.324 等，BF16 舍入级别）。
- 梯度（FP32 参照，12 × 120）：原生 BF16 自身在 RMSNorm 权重上已有最高 57% 的相对误差；Liger 最高 54%，逐参数误差比的中位数 0.96。差异在 BF16 噪声范围内，未见 kernel 错误。
- 单元测试 `tests/test_liger_kernels.py`（FP32，输出与全部梯度对齐到 1e-4）通过。

## 推算

029 每个 microbatch 前向+反向 0.533 s（8 卡、主机下发受限），若文本解码器省下的 0.096 s 全部兑现，每步约 1.09 → 0.89 s（约 1.2×）。这只是推算，以 03-liger-acp 的实测为准。

证据：`attempts/001/microbench.json`、`attempts/001/grad-reference.json` 及对应日志。
