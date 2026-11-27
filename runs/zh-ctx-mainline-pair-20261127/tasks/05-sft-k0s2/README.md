# 05 第二个种子的 K0（SFT → 挖题 → GRPO → 评测）

- **为什么**：K0 那一轮的判定 1 CI 下限只有 +0.19（README“风险”第一条：贴着门槛时再跑一个不同种子的 K0 估运行间波动）。
- **输入**：`configs/train-k0s2.yaml`、`configs/train-data-k0s2.yaml`（`scripts/make_configs.py sft k0s2`）；和 K0 只差训练种子 42 → 43、数据种子 20261106 → 20261128、run 名后缀 -seed2、记录目录和端口。GRPO 配置在节点上由 `make_configs.py grpo START k0s2` 生成（同 K0，换种子；计划在 k0s2 的 SFT 上重挖，挖题 epoch +100）。
- **执行**：节点 pt-bije92x2（10.119.16.248），`scripts/acp-node.sh k0s2`（sft → configs → mining → grpo → eval），产物分别在 05-sft-k0s2、06-mining-k0s2、07-grpo-k0s2；评测名 k0s2grpo-avg3 / start-k0s2，交付模型名 k0s2grpo。W&B 走 17920 转发（11-27 21:0xZ 加入该节点 IP，探测 200）。
- **完成条件**：各段退出码 0、W&B 远端 finished；之后 `compare_delivery.py k0s2grpo kgrpo`、`compare_delivery.py k0grpo k0s2grpo`，gate.py 换成 k0s2grpo 再算一遍。
- **进展**：
  - 11-27 21:05Z 放进 inbox（`ctx-k0s2.sh`，等 8 张卡全空）。
  - attempt 001 失败（与运行无关的冲突）：22:05Z 卡全空的那一刻正好是 TS 线 t29-gate10k 两段之间的空档，SFT 启动后 t29 的下一对 vLLM 服务又占上 GPU 0/1（各约 40 GB），rank 1 在 22:06Z 显存不足退出。22:17–22:18Z 只停了自己的进程树（acp-node.sh k0s2 的子进程），t29 的两个服务没受影响；已告知 71。W&B run `…-codev1-seed2-001` 会显示失败。
  - 22:18Z 重新排队 `ctx-k0s2-retry.sh`：先等 `done/t29-gate10k.exit` 出现，再等 8 张卡连续 5 分钟全空，然后 `acp-node.sh k0s2`（SFT 为 attempt 002）。

<!-- execution-status -->

执行状态：failed；质量状态：未核实。

开始：2026-11-27T22:05:13.944932+00:00；结束：2026-11-27T22:17:04.799735+00:00；退出码：1。

对象存储：已同步到 `whai:open-audio-llm/runs/zh-ctx-mainline-pair-20261127/tasks/05-sft-k0s2/attempts/001`。

<!-- /execution-status -->
