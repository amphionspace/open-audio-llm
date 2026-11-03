# 执行记录

[生效配置](effective.yaml)、[状态](status.json)、[日志](logs/)、[产物](artifacts/)。

<!-- execution-status -->

执行状态：failed；质量状态：未核实。

开始：2026-10-30T09:15:59.868151+00:00；结束：2026-10-30T21:58:25.398423+00:00；退出码：1。

对象存储：已同步到 `whai:open-audio-llm/runs/zh-diar-aug-acp-20261030/tasks/02-train-h/attempts/001`。

<!-- /execution-status -->

## 关于 failed / 退出码 1

训练本身跑完了：`logs/process.log` 有完整的 `12000/12000`、`train_runtime` 44516 s、
`train_loss` 0.279 和 `last_model_checkpoint: .../artifacts/training/checkpoint-12000`，
`status.json` 里训练进程自己的 `process_exit_code` 是 0。

退出码 1 来自收尾时的 W&B 远端结束状态核验（`status.json` 的 `tracking: 远端结束状态未核实`）：
21:30 UTC 起 `api.wandb.ai` 按出口 IP 返回 Cloud Armor 403，ACP 节点传不上去，
远端 run 停在 11455 步、状态 crashed（定位见 `../../../../tracking/wandb-egress-probe.txt`）。

23:0x UTC 在开发机上把本地 run 目录复制到可写位置后用
`wandb sync --no-skip-online`（W&B 流量走海外出口）补录，远端核验结果：
run `AmphionASR-zh-diar-r2-h-speed-specaug-from-emilia2-6k-001`
state `finished`、`global_step` 12000、`launcher/exit_code` 0。
