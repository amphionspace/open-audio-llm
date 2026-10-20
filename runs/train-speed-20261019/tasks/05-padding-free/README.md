# 05-padding-free：文本解码器打包（去掉全部填充计算）

- 输入：04-length-key 实测的 8000 条 token 长度；026 checkpoint-25000 的真实权重；分支代码 `--packed_text`（`integrations/ms_swift/packed_text.py`）。
- 环境：`oal-train` 新装 flash-attn 2.8.3。官方预编译 wheel 只覆盖到 torch 2.9，这里是源码编译（nvcc 12.9，只编 sm_80，`--no-build-isolation --no-deps`，约 18 分钟），没有改动其他包。
- 做法：collate 后把各条有效 token 拼成一行，生成每条从 0 开始的 `position_ids`，并一次性算好 `cu_seq_lens`（避免 transformers 在 28 层里每层都从 position_ids 重算并与主机同步）。只把文本解码器切到 FA2 varlen，音频塔仍用 SDPA。每条样本第一个 token 的标签置为 -100，与填充布局下被平移丢掉的位置一致。
- 进展：attempt 001 完成（2026-10-19）。本机卡被他人占用变化快，前三次运行因卡上显存被占满而失败（`run-*-oom-*.log`），最后在 GPU 7（他人训练占用 75–100%）跑完。

## 结果

**计时**（`padding_free_bench.py`：24 个真实批次，填充 21%，各组逐次交替运行；卡被共享，看相对比例）

| 文本解码器前向+反向 | Liger 开 | Liger 关 |
|---|---|---|
| SDPA + 填充（现状） | 1.00× | 1.00× |
| FA2 + 填充 | 0.98× | 0.96× |
| FA2 打包 | 1.35×（中位 1.27×） | 1.35×（中位 1.36×） |

**数值**
- 文本解码器、真实权重、FP32 参照（`padding_free_grad_reference.py`）：隐藏状态误差 SDPA 0.93% / 打包 0.90%；梯度误差中位数 4.7% / 4.4%。
- 完整训练链路（`packed_forward_check.py`：真实样本 → ms-swift collator → 打包 → 完整模型含音频批处理、Liger、有监督头）：打包与"填充 + FA2"的 loss、梯度完全相同；相对 FP32 的梯度范数误差，FA2 为 0.96%，现状 SDPA 为 8.4%。
- 单元测试 `tests/test_packed_text.py`：CPU 打包逻辑（左右填充）、拒绝无法对齐的逐 token 字段、GPU 上打包与填充的输出和梯度对齐。

## 结论

- 打包本身不引入误差；FA2 的数值比现状 SDPA 更接近 FP32。
- 文本解码器约快 1.3×。与 Liger 叠加后，按单卡计时推算整步约快 1.35–1.4×，以 ACP 端到端核对为准（06-packed-acp，待用户放行）。
- 打包在批次层面完成，不改变采样和批次组成，没有排序键方案那样的组成副作用。

证据：`attempts/001/bench-liger.json`、`bench-native.json`、`grad-reference.json`、`full-path-check.json` 及日志。
