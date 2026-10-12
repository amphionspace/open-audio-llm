# 组装本轮训练数据配置

输入：上一轮 `01-prepare` 执行 004 的数据配置（旧来源、排除与 clean 规则）、NOTSOFAR/CHiME-6/RAMC 窗口 Catalog、词级 6–12 人合成 Catalog。

完成条件：各组权重与配方一致；NOTSOFAR 按设备数分区；词级合成按语言 × 人数拆分；合并 Catalog 与 roots 可读；CPU 数据预检通过。

进展：执行 001 因合并 Catalog 中同一数据集存在多个版本而误判重复失败，改为按（数据集，版本）去重后，执行 002 通过，94 个训练来源。

证据：`attempts/002/artifacts/report.json`、`train-data.yaml`、`catalog.jsonl`、`subsets/`；预检见实验 `tracking/data-preflight.json`。
