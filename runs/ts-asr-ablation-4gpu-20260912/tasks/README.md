# 目标说话人识别：任务导航

输入与完成条件沿用原方案和配置；下表链接到保留原位的记录。

| 工作 | 入口与证据 |
|---|---|
| 执行入口 | [evaluate_checkpoint.py](../evaluate_checkpoint.py)、[experiment_callback.py](../experiment_callback.py)、[launch.py](../launch.py)、[prepare.py](../prepare.py)、[summarize.py](../summarize.py) |
| 配置与方案 | [asr-dev.yaml](../asr-dev.yaml)、[asr-expanded-dev.yaml](../asr-expanded-dev.yaml)、[ts-dev.yaml](../ts-dev.yaml) |
| 结果与证据 | [preflight-verification.json](../preflight-verification.json)、[status.json](../status.json) |
| 产物目录 | [control](../control)、[encoder_lr_5x](../encoder_lr_5x)、[english_ts_positive](../english_ts_positive)、[language_60_40](../language_60_40)、[saved-main](../saved-main) |

进展与结论：历史状态未重新核实，不能根据目录或 PID 文件判定正在运行。有 summary、verification 时以其中原始结果为依据；没有结果时不视为通过。
