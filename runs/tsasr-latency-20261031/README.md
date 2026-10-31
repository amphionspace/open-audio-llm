# TS-ASR 出字速度（2026-10-31）

目标：在不改线上部署的前提下，量化提升 NVIDIA 线上（Focus，RTX PRO 4500 Blackwell 32 GiB，4 路分时共享）TS-ASR 出字速度的各个手段，给出"收益 / 代价 / 风险"，由用户决定上线哪些。

- 模型：10.28 版本交付包（054 avg3），`/workspace/data/packages/amphionasr-1.7b-v3-tsrobust-054avg3`。
- 节奏与请求：照 audiollm-go origin/main `0966b463`（5 s 注册、6 s 稳定窗口复用、1.5 s 预览节奏、27 s final 窗口、输出上限 128）。
- 本机 A800 被他人任务占满，延迟只比较同卡上的相对差值；线上绝对数字需在 Focus test 复测。

| 任务 | 内容 |
|---|---|
| [01-stream-bench](tasks/01-stream-bench/README.md) | 各服务变体（base / suffix / fp8 / fp8+suffix / base 复测）的流式回放与 prefill 微基准 |
| [02-accuracy-gates](tasks/02-accuracy-gates/README.md) | suffix、fp8、线上 "5-second" 指令三项的 v12 全量准确率闸门（AmphionEval） |

<!-- execution-status -->

执行状态：partial；质量状态：未核实。

开始：未知；结束：未结束；退出码：未知。

<!-- /execution-status -->
