# 实验与分析记录：任务导航

输入与完成条件沿用原方案和配置；下表链接到保留原位的记录。

| 工作 | 入口与证据 |
|---|---|
| 执行入口 | [bmm_probe.py](../bmm_probe.py)、[compare_ddp.py](../compare_ddp.py)、[encoder_benchmark.py](../encoder_benchmark.py)、[encoder_setup.py](../encoder_setup.py)、[eval_fp32.py](../eval_fp32.py)、[final-audio-batching.py](../final-audio-batching.py)、[full_update_benchmark.py](../full_update_benchmark.py)、[precision_gold.py](../precision_gold.py)、[precision_probe.py](../precision_probe.py)、[prepare_batches.py](../prepare_batches.py)、[run_ddp_benchmark.py](../run_ddp_benchmark.py)、[serial_front_probe.py](../serial_front_probe.py)、[stage_probe.py](../stage_probe.py) |
| 配置与方案 | 未发现根层记录 |
| 结果与证据 | [summary.json](../summary.json) |
| 产物目录 | [batched-base-asr](../batched-base-asr)、[ddp-batched](../ddp-batched)、[ddp-native](../ddp-native)、[fp32-base-asr](../fp32-base-asr)、[posttrain-batched-asr](../posttrain-batched-asr)、[posttrain-native-asr](../posttrain-native-asr) |

进展与结论：历史状态未重新核实，不能根据目录或 PID 文件判定正在运行。有 summary、verification 时以其中原始结果为依据；没有结果时不视为通过。
