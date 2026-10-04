# 目标说话人识别：任务导航

输入与完成条件沿用原方案和配置；下表链接到保留原位的记录。

| 工作 | 入口与证据 |
|---|---|
| 执行入口 | [evaluate.py](../evaluate.py)、[evaluate_checkpoint.py](../evaluate_checkpoint.py)、[preflight.py](../preflight.py)、[prepare_recipe.py](../prepare_recipe.py)、[prepare_resume.py](../prepare_resume.py)、[resume_state.py](../resume_state.py)、[run_train.py](../run_train.py)、[sample_gpu.py](../sample_gpu.py)、[start.py](../start.py)、[status.py](../status.py)、[training_audit.py](../training_audit.py)、[verify_startup.py](../verify_startup.py) |
| 配置与方案 | [asr-dev.yaml](../asr-dev.yaml)、[asr-expanded-dev.yaml](../asr-expanded-dev.yaml)、[train-data.yaml](../train-data.yaml)、[ts-dev.yaml](../ts-dev.yaml) |
| 结果与证据 | [startup-verification.json](../startup-verification.json)、[status.py](../status.py) |
| 产物目录 | [base-ts-dev](../base-ts-dev)、[initial-ts-dev](../initial-ts-dev)、[training](../training) |

进展与结论：历史状态未重新核实，不能根据目录或 PID 文件判定正在运行。有 summary、verification 时以其中原始结果为依据；没有结果时不视为通过。
