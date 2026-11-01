# TS-ASR 出字速度（2026-10-31）

目标：量化提升 NVIDIA 线上（Focus，RTX PRO 4500 Blackwell 32 GiB，4 路分时共享）TS-ASR 出字速度的各个手段，给出收益 / 代价 / 风险，由用户决定上线哪些。

- 模型：10.28 版本交付包（054 avg3）。节奏与请求照 audiollm-go origin/main `0966b463`（5 s 注册、6 s 稳定窗口复用、1.5 s 预览节奏、27 s final 窗口、输出上限 128）。
- 本机 A800 被他人任务占满（分时共享，和线上相同机制）；延迟只看同卡相对差值，线上绝对数字需在 Focus test 复测。
- W&B：出口 10-30 起被封，先离线记录，10-31 经海外代理补录并核验远端（10 个 run，状态 finished）。

## 结论

| 手段 | 延迟收益（本机 16 / 48 路） | 准确率（v12 全量或配对） | 代价 | 建议 |
|---|---|---|---|---|
| **编码器去 GPU 同步**（`fix/tsasr-encoder-host-syncs`） | 预览滞后 p50 −74% / −69%，断句到 final p50 −53% / −60%，p95 −67% / −72%；单路 prefill 199→71 ms；48 路吞吐 +20% | 200 条完整解码逐字相同；配对 final 一致率与同版本重跑相同 | 改 runtime 3 处，无新依赖 | 上线（交付包 + audiollm-go vendor） |
| **suffix 投机解码** | 预览滞后 p50 −26%，断句到 final p50 −23% / −38%，p95 −42% / −44%；接受率 0.61–0.68 | v12：CER 13.04 vs 13.07，翻转 40（噪声 50） | 新依赖 arctic-inference 0.1.1；关闭异步调度 | 上线前在 Focus test 复测（与去同步叠加后的增量未测） |
| 静音期提前开始 final（runtime） | 断句到 final p50 −0.4~0.7 s | final 输入不变 | 说话人继续说时白算的 GPU 未计入 | 建议 runtime 实现后复测 |
| final 复用预览窗口（runtime） | −0.1~0.4 s | final CER 约翻倍（7.5→14.6%，11.5→19.8%） | — | 否决 |
| FP8 权重量化（Marlin，A800 上唯一可跑的路径，只量化权重） | 无收益，48 路更慢 | CER 13.14，翻转 80（噪声 50） | 不代表 H20：H20 的 FP8 权重和激活都量化，见下一行 | 不建议（需目标卡另测） |
| FP8（H20 数值，A800 模拟，任务 07） | 没测，要真 H20 | v12 CER 13.08，翻转 99（噪声 50）；ASR 全量 4.292→4.301；警务术语少命中 2–7 个 | 未含 FP8 KV cache | 取舍见任务 07 |
| 缩短 TS 指令（39 token） | 单路 242 vs 240 ms，16 路 715 vs 718 ms：测不出 | — | 需重训 + 协议变更 | 不做 |
| 线上 "5-second" 指令（训练为 "3-second"） | — | v12：CER 13.06，翻转 45：与训练指令无差别 | — | 不用改 |

根因：每个请求的音频编码有约 30 次 GPU→CPU 同步（编码器每层一次 `max_seqlen.item()`，加 `sep_token.py` 里长度的 `.tolist()`/`.item()`），分时共享的 GPU 上每次都要等其他进程的时间片（05 剖析：GPU kernel 每请求仅约 10 ms）。

## 任务

| 任务 | 内容 |
|---|---|
| [01-stream-bench](tasks/01-stream-bench/README.md) | 服务变体 base / suffix / fp8 / fp8+suffix / base 复测的流式回放（其中 prefill 对比因处理缓存无效，见 04） |
| [02-accuracy-gates](tasks/02-accuracy-gates/README.md) | suffix、fp8、"5-second" 指令的 v12 全量闸门，`results/compare.json` |
| [03-runtime-levers](tasks/03-runtime-levers/README.md) | 静音期提前 final、final 复用预览窗口（attempt 001 因处理缓存偏差中止，以 002 为准） |
| [04-prefill-bench](tasks/04-prefill-bench/README.md) | 指令长度与 CPU/编码器开销的 prefill 微基准 |
| [05-profile-prefill](tasks/05-profile-prefill/README.md) | torch profiler 定位编码器同步点 |
| [06-nosync-ab](tasks/06-nosync-ab/README.md) | 去同步 runtime 与原交付包交替各两轮的 A/B |
| [07-fp8-h20-emu](tasks/07-fp8-h20-emu/README.md) | H20 数值的 FP8 量化损失：七项全量集，配 bf16 重跑作噪声参照 |

<!-- execution-status -->

执行状态：completed；质量状态：未核实。

开始：未知；结束：未结束；退出码：未知。

<!-- /execution-status -->
