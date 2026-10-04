# 数据清洗：任务导航

输入与完成条件沿用原方案和配置；下表链接到保留原位的记录。

| 工作 | 入口与证据 |
|---|---|
| 执行入口 | [analyze_pilot.py](../analyze_pilot.py)、[check_moss_wheel.py](../check_moss_wheel.py)、[check_stages.py](../check_stages.py)、[moss_batch.py](../moss_batch.py)、[moss_batch_fast.py](../moss_batch_fast.py)、[moss_batch_multi.py](../moss_batch_multi.py)、[moss_batch_probe.py](../moss_batch_probe.py)、[moss_probe.py](../moss_probe.py)、[pilot.py](../pilot.py)、[prepare_full.py](../prepare_full.py)、[prepare_pilot.py](../prepare_pilot.py)、[prepare_purity_probe.py](../prepare_purity_probe.py)、[production.py](../production.py)、[production_dual_gpu.py](../production_dual_gpu.py)、[production_fast.py](../production_fast.py)、[publish_pilot.py](../publish_pilot.py)、[qwen_queue.py](../qwen_queue.py)、[report_pilot.py](../report_pilot.py)、[stages.py](../stages.py)、[transcript_parser.py](../transcript_parser.py)、[verify_runs.py](../verify_runs.py)、[verify_wandb.py](../verify_wandb.py) |
| 配置与方案 | [moss-environment-plan.log](../moss-environment-plan.log) |
| 结果与证据 | [catalog-export-verification.json](../catalog-export-verification.json)、[moss-batch-verification.json](../moss-batch-verification.json)、[moss-probe-summary.json](../moss-probe-summary.json)、[stage-verification](../stage-verification)、[stage-verification.json](../stage-verification.json)、[wandb-production-verification.json](../wandb-production-verification.json)、[wandb-verification.json](../wandb-verification.json) |
| 产物目录 | [clean-sources](../clean-sources)、[full](../full)、[inputs](../inputs)、[multi-gpu-20260924](../multi-gpu-20260924)、[performance-20260924](../performance-20260924)、[pilot-en](../pilot-en)、[pilot-zh](../pilot-zh)、[purity-probe](../purity-probe)、[qwen-ready](../qwen-ready)、[snapshot](../snapshot)、[stage-verification](../stage-verification) |

进展与结论：历史状态未重新核实，不能根据目录或 PID 文件判定正在运行。有 summary、verification 时以其中原始结果为依据；没有结果时不视为通过。
