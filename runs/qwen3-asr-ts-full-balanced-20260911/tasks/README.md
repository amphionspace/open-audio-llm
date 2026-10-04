# 目标说话人识别：任务导航

输入与完成条件沿用原方案和配置；下表链接到保留原位的记录。

| 工作 | 入口与证据 |
|---|---|
| 执行入口 | [audit_data.py](../audit_data.py)、[eval_model.py](../eval_model.py)、[finalize.py](../finalize.py)、[restart_evaluation.py](../restart_evaluation.py)、[restart_watcher.py](../restart_watcher.py)、[run_train.py](../run_train.py)、[status.py](../status.py)、[watch_checkpoints.py](../watch_checkpoints.py) |
| 配置与方案 | [asr-dev.yaml](../asr-dev.yaml)、[asr-expanded-dev.yaml](../asr-expanded-dev.yaml)、[data.yaml](../data.yaml)、[test-data.yaml](../test-data.yaml)、[train-data.yaml](../train-data.yaml)、[ts-dev.yaml](../ts-dev.yaml) |
| 结果与证据 | [status.py](../status.py) |
| 产物目录 | [base-asr-dev](../base-asr-dev)、[base-asr-expanded-dev](../base-asr-expanded-dev)、[base-ts-dev](../base-ts-dev)、[diagnostic-old500-cached](../diagnostic-old500-cached)、[evaluations](../evaluations)、[smoke](../smoke)、[training](../training) |

进展与结论：历史状态未重新核实，不能根据目录或 PID 文件判定正在运行。有 summary、verification 时以其中原始结果为依据；没有结果时不视为通过。
