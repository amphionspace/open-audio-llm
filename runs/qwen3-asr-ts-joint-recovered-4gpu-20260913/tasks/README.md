# 目标说话人识别：任务导航

输入与完成条件沿用原方案和配置；下表链接到保留原位的记录。

| 工作 | 入口与证据 |
|---|---|
| 执行入口 | [cleanup_storage.py](../cleanup_storage.py)、[evaluate.py](../evaluate.py)、[evaluate_checkpoint.py](../evaluate_checkpoint.py)、[prepare.py](../prepare.py)、[prepare_save_policy.py](../prepare_save_policy.py)、[recovery_audit.py](../recovery_audit.py)、[run_train.py](../run_train.py) |
| 配置与方案 | [asr-dev.yaml](../asr-dev.yaml)、[asr-expanded-dev.yaml](../asr-expanded-dev.yaml)、[storage-cleanup-plan.json](../storage-cleanup-plan.json)、[train-data.yaml](../train-data.yaml)、[ts-dev.yaml](../ts-dev.yaml) |
| 结果与证据 | [model-only-verification.json](../model-only-verification.json) |
| 产物目录 | [resume](../resume)、[training](../training)、[training-model-only](../training-model-only) |

进展与结论：历史状态未重新核实，不能根据目录或 PID 文件判定正在运行。有 summary、verification 时以其中原始结果为依据；没有结果时不视为通过。
