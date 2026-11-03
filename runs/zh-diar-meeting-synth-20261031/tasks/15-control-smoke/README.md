# 15 对照训练 GPU0 小试

## 输入

- 配置 `configs/train-control-smoke.yaml`，由 `scripts/make_control_configs.py` 的 `smoke_config()` 从 `configs/train-control.yaml` 生成：数据配方、起点模型和训练参数不变，只改成本机单卡（001/002 用 GPU 0，003 用 GPU 5）、单进程、不做梯度累积、40 步、不保存。每步样本数由数据配置的 `batching`（`max_samples: 2`、`max_duration: 650`）决定，与正式训练每卡每个微批相同；`--per_device_train_batch_size` 不起作用。
- 代码快照 `shared/code-control-v2`：本分支合并 origin/main（6d5a25e）后的提交 1fea5cb 加上未提交的 `meeting_online.py` 改动，`SOURCE.txt` 记来源。2026-11-03 再次合并 main 后，生成器把配置改指 `code-control-v3`。001–003 实际用的是 v2，以各次的 `effective.yaml` 为准；v3 的改动只在数据渲染路径上，由 `tracking/preflight-control-local-v3.json` 覆盖，没有重跑小试。
- 用户 2026-11-02：“你可以用GPU0先小试一下，或者将main merge过来确保以后提pr没问题”。

## 完成条件

合并后的代码在在线会议数据上能端到端训练：模型加载、数据预检通过，loss 有限且量级合理，数据加载等待不拖慢步速，W&B 远端有训练指标。这是可行性检查，不产出结果模型。

## 进展

- attempt 001（GPU 0）：训练回调 `many_speaker_training_audit` 断言 节点 × 进程 = 8，小试配置没有 `cluster` 段，开训即 `KeyError: 'cluster'`。修复：期望值改由配置 `parameters.expected_world_size` 给出（对照训练 8、小试 1），无 `cluster` 段按 1 个节点算。
- attempt 002（GPU 0）：第 1 步前向、反向和回调检查都通过，在优化器首步建 Adam 状态时 OOM。原因是期间 GPU 0 上来了 062 评测服务（14.7 GB）和其他用户的进程，共占 52 GB。
- attempt 003（GPU 5，当时空闲 45 GB）：完成，退出码 0。

## 结果

执行完成，可行性检查通过（attempt 003）：
- 回调检查通过：2,038,053,504 个参数全部可训练，没有 LoRA，数据全覆盖，学习率大于 0（`actual-training-contract.json`、`actual-learning-rates.json`）。
- 40 步 loss 都是有限值，范围 0.12–0.88。含合成会议样本的步中位数约 0.6，其他步约 0.3。合成会议在 80 个样本里占 18 个（22.5%），配比是 24.7%。
- grad_norm 中位数约 8，第 18 步最大 111，被 `max_grad_norm: 1.0` 裁剪。小批量下出现尖峰在预期内，正式训练的全局批量是 32。
- 数据加载跟得上：第 1 步等 worker 启动 13.5 s，之后取数时间只占墙钟的 0.5%。每个样本准备耗时中位数 0.26 s，最慢 9.9 s。
- 显存峰值 39.5 GiB，这是正式训练每卡同样大小的微批（2 个样本、≤650 s）的单卡值。
- W&B `AmphionASR-zh-diar-synth-control-smoke-gpu0-003` 远端状态 finished，`train/loss` 40 个点与本地一致。

注意：这次只跑了单卡、40 步，不能说明 8 卡下的步速、显存和长时间稳定性，这些在正式训练里看。

## 证据

`attempts/003/status.json`、`attempts/003/logs/process.log`、`attempts/003/artifacts/training/{logging.jsonl,performance-rank0.jsonl,actual-training-contract.json}`、`attempts/003/artifacts/data-preflight.json`；001、002 的日志保留原样。
