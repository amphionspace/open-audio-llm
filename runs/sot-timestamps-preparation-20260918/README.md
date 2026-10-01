# 多说话人转写：sot-timestamps-preparation-20260918

<!-- experiment-navigation -->

这是历史实验，产物保留原位。输入、完成条件、进展和结果见[任务导航](tasks/README.md)及原始方案；本次仅补齐导航，未重跑或重判质量。

执行状态：待核实；质量状态：沿用已有证据，缺少记录时保持未核实。

## 原始概览

# 时间戳实验准备状态

分支：`feat/sot-next-stage`。尚未启动训练或运行新模型评测。

已冻结 `../AmphionData/results/sot-source-alignment-20260918` 的 270 个对齐分片，
索引 `source-alignments.sqlite` 含 549,056 条英文 train 源片段：418,300 条 aligned、
130,756 条 needs_review。来源、模型配置及校验和见 `source-alignments.json`。
原始混音、文字、train/dev/test 划分均未改写。

`train-data.yaml`、`sot-dev.yaml`、`asr-dev.yaml` 保留原 62 个 SOT 条件与 60/30/10
回放比例、KL=2；这些是完整实验配置，尚不具备全量启动条件。

诊断抽样为每个条件第一分片的前 8 条，不是正式评测样本，不用于报告模型效果：

| split | 可构建时间目标 | 涉及待复核对齐 | 缺少源对齐 |
| --- | ---: | ---: | ---: |
| train | 69 | 123 | 304 |
| dev | 0 | 0 | 496 |
| test | 0 | 0 | 496 |

完整分条件计数见 `coverage-audit.json`。先遇到的错误作为该样本原因，因此不是所有
缺失源片段的穷尽统计。来源计划仅选择英文 train，中文缺失源于上游将“clean 只有
 test”当作跳过原始 train 的依据。此规则与本项目“没有适用 clean 训练版本时允许
原始训练集”的约定不同；补齐时应对应既有混音实际使用的音频和版本，不能只换标注。

`ready-examples.jsonl` 的 5 条真实英文混音覆盖 1～5 人。已通过音频读取、完整文本
目标构建、时间轴一致性、原生消息转换及参考自评分检查；见 `smoke-data-report.json`。
自评分用于检查代码，不是模型成绩。强制启用 0.5 倍速后，时间戳样本仍保持原音频
时长；普通 ASR 回放维持既有增强行为。

下一步依赖：补齐中文来源及 dev/test 的真实对齐，处理 needs_review 后冻结新索引。
不得通过删掉缺失发言、替换整段边界或筛选固定 dev/test 来绕过缺口。
启动方案：从上一轮 checkpoint-60000 初始化，两卡全参数试训 2,000 步，保留原
学习率（encoder/aligner 2e-5、LLM 1e-5）、回放和 teacher；新优化器、新 warmup，
只保存模型。正式启动前用新输出协议生成固定开发集基线，并接上定期普通 ASR 保持
评测和 W&B。操作入口见 `examples/train/qwen3-asr/SOT.md`。
