# 多说话人转写：任务导航

输入与完成条件沿用原方案和配置；下表链接到保留原位的记录。

| 工作 | 入口与证据 |
|---|---|
| 执行入口 | [evaluate_checkpoint.py](../evaluate_checkpoint.py)、[evaluate_sot.py](../evaluate_sot.py)、[launch.sh](../launch.sh)、[prepare_trial.py](../prepare_trial.py)、[repair_float_record.py](../repair_float_record.py)、[training_audit.py](../training_audit.py) |
| 配置与方案 | [asr-dev.yaml](../asr-dev.yaml)、[sot-dev-original.yaml](../sot-dev-original.yaml)、[train-data.yaml](../train-data.yaml) |
| 结果与证据 | [startup-verification.json](../startup-verification.json) |
| 产物目录 | [baseline-asr](../baseline-asr)、[baseline-sot](../baseline-sot)、[data](../data)、[partial-cache](../partial-cache)、[starting-asr](../starting-asr)、[training](../training)、[wandb-sync](../wandb-sync) |

进展与结论：历史状态未重新核实，不能根据目录或 PID 文件判定正在运行。有 summary、verification 时以其中原始结果为依据；没有结果时不视为通过。
