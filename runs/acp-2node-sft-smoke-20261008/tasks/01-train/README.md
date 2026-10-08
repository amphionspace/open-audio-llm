# 01-train：2 节点 LoRA 冒烟训练

- 输入：`configs/train.yaml`、`configs/data.yaml`；模型 `/workspace/model/Qwen3-ASR-1.7B`。
- 完成条件：两个节点都加入同一个 torchrun（world size 16），跑完 30 步；W&B run 远端核验 finished；checkpoint-10/20/30 上传 whai，本地副本删除；`checkpoint-handoff.json` 指向远端 checkpoint-30。
- 进展：本机预检通过，Catalog 读取 2400 条；已提交 ACP 排队（任务号见执行记录 `status.json` 的 `cluster.job`）。
- 结果与证据：执行完成后补充。

<!-- execution-status -->

执行状态：submitted；质量状态：未核实。

开始：未知；结束：未结束；退出码：未知。

<!-- /execution-status -->
