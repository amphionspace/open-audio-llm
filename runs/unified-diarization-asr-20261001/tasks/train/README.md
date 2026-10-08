# 统一格式训练

输入：[训练配置](../../configs/train-unified.yaml) 和 [固定数据配置](../../shared/data/unified-train.yaml)。初始化自固定初始 checkpoint，使用两张 GPU、全参数训练、新 optimizer 和 scheduler。

完成条件：1000 步训练结束并保存 checkpoint、退出码和 W&B 远端核验证据。质量结论需后续固定集评测。

进展：预备数据预检通过；实际启动时重新核验冻结配置。

运行：`open-audio-llm experiment run --config runs/unified-diarization-asr-20261001/experiment.yaml --task train`。

证据：[预备预检](../../shared/data-preflight-prelaunch.json)、[逐次执行记录](attempts/)。

<!-- execution-status -->

执行状态：failed；质量状态：未核实。

开始：2026-10-02T06:08:30.811780+00:00；结束：2026-10-02T08:45:48.421652+00:00；退出码：1。

<!-- /execution-status -->
