# 10 后续 3 训练臂 A（real），判断项用大批 TS 样本

- **输入**：同任务 04，只差判断项（README“后续 3：判断项用大批 TS 样本”）：每卡每步另取 16 条 TS 回放样本，回放损失 = token 平均 KL + 1.0 × 它们判断 token KL 的均值；配置 `configs/train-real-decbatch.yaml`。
- **完成条件**：
  - 训练跑完，`/workspace/data/ae-runs/ctx-kl-anchor-20261126/real-decbatch/` 下有一半处和最后的 checkpoint；
  - W&B 远端 run 为 finished，并能看到 `train/ce_asr`、`train/replay_kl` 到最后一步；`replay_kl_decision`、`decision_batch_kl` 训练后补录；
  - 生效核查（README“后续 3”）：第 1 步和任务 04 相同；第 300–500 步回放流 TS 负样本判断 token KL 不高于任务 04 同窗口的上限；最后 10% ≤ 0.010。
- **进展**：
  - attempt 001（11-27 11:34–12:00Z，ACP pt-dalkekon，节点 10.119.16.251）：手动中断（interrupted，退出码 143）。
    - 生效核查第 1 条通过：第 1 步与任务 04 是同一个 batch，CE 0.1708、token 平均 KL 0.00019 相同；判断项 KL 1.6e-4，只有 fp32 / bf16 的数值差，不为 0。
    - 中断原因：每步约 10 s（预估约 4 s）。节点上判断流的 4 个数据 worker 占满，GPU 间歇空闲，128 核里 77% 空闲，是数据读取跟不上（节点上约 1.5 s/条，开发机测的是 0.22 s/条）。
    - 修正（6209b18）：`decision_workers` 4 → 12。只影响读取吞吐，样本顺序和在线混合不变。
  - attempt 002（11-27 12:02Z 起，同一节点）：按写定的规则中途停止（interrupted，退出码 143，13:29Z），停在第 600 步，每步约 7–8.7 s。
    - 第 1 步仍与任务 04 相同。
    - W&B：中断后远端为 crashed（同步进程随训练一起停）。`train/ce_asr` 62 行，`train/replay_kl_decision`、`train/decision_batch_kl` 各 63 行，都已上传，不用补录。
    - **生效核查第 2 条没过**（`scripts/replay_kl_window.py`，日志 `midrun-check.log`）：第 300–500 步的 21 个相同窗口里，回放流 TS 负样本判断 token KL 为 0.0389，高于任务 04 的上限 0.0181（21 个窗口里 20 个更高）；后续 2 同窗口为 0.1005。
    - 判断项样本自己的判断 KL（训练用的那 64 条 / 步）和回放流的留出估计差不多，说明不是过拟合，而是这一项在训练集上也没能把判断 KL 压到后续 1 以下。
- **结果 / 证据**：执行记录在 `attempts/<编号>/`（status.json、W&B 核验、`artifacts/training/logging.jsonl`、`replay-kl-by-task.jsonl`）。

<!-- execution-status -->

执行状态：interrupted；质量状态：未核实。

开始：2026-11-27T12:02:27.131202+00:00；结束：2026-11-27T13:29:40.894322+00:00；退出码：143。

<!-- /execution-status -->
