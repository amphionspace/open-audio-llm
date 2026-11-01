# 03 runtime 侧手段

- 输入：同 01，10.28 版本交付包原样服务（base）。
- 方法：`scripts/stream_bench.py` 的四个对照臂按条目配对轮流：main（现状）、prepare（静音 300 ms 提前开始 final）、windowed（final 复用预览稳定窗口）、windowed-prepare。
- 完成条件：每个臂的断句到 final、final 解码时间、final 字错率与缺席误输出率（1/16/48 路）。
- 局限：prepare 是预计算全部命中的上限，没计入说话人继续说时白算的 GPU；回放不经过 gateway，不含调度器限额与网络。

<!-- execution-status -->

执行状态：completed；质量状态：未核实。

开始：2026-10-31T17:32:46.888427+00:00；结束：2026-10-31T17:52:39.446628+00:00；退出码：0。

<!-- /execution-status -->
