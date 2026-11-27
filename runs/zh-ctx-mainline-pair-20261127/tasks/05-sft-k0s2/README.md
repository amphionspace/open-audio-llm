# 05 第二个种子的 K0（SFT → 挖题 → GRPO → 评测）

- **为什么**：K0 那一轮的判定 1 CI 下限只有 +0.19（README“风险”第一条：贴着门槛时再跑一个不同种子的 K0 估运行间波动）。
- **输入**：`configs/train-k0s2.yaml`、`configs/train-data-k0s2.yaml`（`scripts/make_configs.py sft k0s2`）；和 K0 只差训练种子 42 → 43、数据种子 20261106 → 20261128、run 名后缀 -seed2、记录目录和端口。GRPO 配置在节点上由 `make_configs.py grpo START k0s2` 生成（同 K0，换种子；计划在 k0s2 的 SFT 上重挖，挖题 epoch +100）。
- **执行**：节点 pt-bije92x2（10.119.16.248），`scripts/acp-node.sh k0s2`（sft → configs → mining → grpo → eval），产物分别在 05-sft-k0s2、06-mining-k0s2、07-grpo-k0s2；评测名 k0s2grpo-avg3 / start-k0s2，交付模型名 k0s2grpo。W&B 走 17920 转发（11-27 21:0xZ 加入该节点 IP，探测 200）。
- **完成条件**：各段退出码 0、W&B 远端 finished；之后 `compare_delivery.py k0s2grpo kgrpo`、`compare_delivery.py k0grpo k0s2grpo`，gate.py 换成 k0s2grpo 再算一遍。
- **进展**：11-27 21:05Z 放进 inbox，等 8 张卡全空（TS 线 t29-gate10k 预计 22:30–22:45Z 结束）。
