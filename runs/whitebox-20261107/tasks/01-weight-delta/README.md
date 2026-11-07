# 01 权重更新分析

- 输入：S = SFT 073 avg3；A = v15 第 400 步 − S；B = v16 第 400 步 − v15 插值；SFT = 073 avg3 − 066 avg3。脚本 `scripts/weight_delta.py`，结果 `delta.json`。
- 结果：GRPO 改了约 16% 的 bf16 参数（音频塔 27%、LLM 20%，embedding / lm_head / norm 基本不动）；相对改动约 6e-4，是 SFT 的 1/15；LLM 第 13–16 层最大；A 与 B 余弦 0.10–0.18（近似正交），与 SFT 余弦约 0.002；抽查矩阵的 90% 能量需要约一半的秩（不是低秩）。
