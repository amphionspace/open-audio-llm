# 目标说话人识别：任务导航

输入与完成条件沿用原方案和配置；下表链接到保留原位的记录。

| 工作 | 入口与证据 |
|---|---|
| 执行入口 | [benchmark_encoder_streams.py](../benchmark_encoder_streams.py)、[benchmark_update.py](../benchmark_update.py)、[eval_paired.py](../eval_paired.py)、[eval_runner.py](../eval_runner.py)、[prepare_metadata.py](../prepare_metadata.py)、[rejected_batched_audio.py](../rejected_batched_audio.py)、[resume_paired.py](../resume_paired.py)、[resume_runner.py](../resume_runner.py)、[summarize_performance.py](../summarize_performance.py)、[verify_audio_batch.py](../verify_audio_batch.py)、[verify_encoder_precision.py](../verify_encoder_precision.py) |
| 配置与方案 | [data.yaml](../data.yaml) |
| 结果与证据 | [audio-batch-summary.json](../audio-batch-summary.json) |
| 产物目录 | [base-batched-dev](../base-batched-dev)、[base-hybrid-dev](../base-hybrid-dev)、[base-parallel-dev](../base-parallel-dev)、[checkpoints](../checkpoints)、[original-1250-dev](../original-1250-dev)、[original-250-dev](../original-250-dev)、[original-final-dev](../original-final-dev)、[paired](../paired) |

进展与结论：历史状态未重新核实，不能根据目录或 PID 文件判定正在运行。有 summary、verification 时以其中原始结果为依据；没有结果时不视为通过。
