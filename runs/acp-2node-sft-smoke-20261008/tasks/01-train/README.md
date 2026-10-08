# 01-train：2 节点 LoRA 冒烟训练

- 输入：`configs/train.yaml`、`configs/data.yaml`；模型 `/workspace/model/Qwen3-ASR-1.7B`。
- 完成条件：两个节点都加入同一个 torchrun（world size 16），跑完 30 步；W&B run 远端核验 finished；checkpoint-10/20/30 上传 whai，本地副本删除；`checkpoint-handoff.json` 指向远端 checkpoint-30。

## 执行记录

| 执行 | 任务号 | 结果 |
| --- | --- | --- |
| 001 | 无 | 提交被拒：`sco --env` 只接受一个变量。已改为在启动脚本里设置（PR #35） |
| 002 | `pt-4vnq1uxw` | 失败，见下 |
| 003 | `pt-ubntzbtp` | 执行完成，功能达标，见下 |

## 002 的发现（2026-10-08）

- 已验证：ACP 启动、rank 0 启动器、W&B 启动核验、两节点 torchrun 建立（`world_size: 16`）、两节点读取 Catalog 2400 条。
- 失败原因 1：rank5 打开共享元数据缓存的锁文件时报 `FileExistsError`。quarkfs 上两台机器同时创建同一个文件，即使没有 O_EXCL 也可能返回 EEXIST。修复：锁文件统一用 `catalog_cache.exclusive_lock`，遇到 EEXIST 就重新打开。
- 失败原因 2：另一节点的 pod 失败退出后，ACP 立即判定任务失败，3 秒内删掉了 master pod，打断了 rank 0 的 W&B 结束核验和对象存储同步。修复：非 0 节点等 rank 0 写出 `cluster-rank0-done.json` 后再退出。
- 补录：W&B run `acp-2node-sft-smoke-20261008-002` 已补标为 failed（`launcher/exit_code=1`，远端核验）；执行记录已用 `storage sync` 上传 whai。

<!-- execution-status -->

执行状态：submitted；质量状态：未核实。

开始：未知；结束：未结束；退出码：未知。

<!-- /execution-status -->

## 003 的结果（2026-10-08）

执行完成（退出码 0，ACP 状态 SUCCEEDED），完成条件全部满足。

| 检查项 | 结果 |
| --- | --- |
| 多机 | `world_size: 16, local_world_size: 8`，两节点同一 torchrun |
| 训练 | 30/30 步，用时 29 秒，train_loss 0.349，最后 loss 0.045 |
| W&B | `acp-2node-sft-smoke-20261008-003` 远端核验 finished，`launcher/exit_code=0` |
| checkpoint | whai 上有 checkpoint-10/20/30；中间 checkpoint 只有权重（8 个文件，无优化器），checkpoint-30 含优化器等完整状态（26 个文件）；本地无 checkpoint 目录 |
| handoff | `checkpoint_remote` 指向 whai 上的 checkpoint-30 |
| 执行记录 | 已同步 whai（54 个文件）；rank 0 写出 `cluster-rank0-done.json` 后任务才结束 |
