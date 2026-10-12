# 中文远场配比的数据配置

在上一轮冻结配置上替换 AISHELL-4/AliMeeting/RAMC 为 v2，加入单麦视角、模拟远场与中文远场回放，回放组缩到 12；其余来源和权重不变，种子 45。

执行 001 因脚本的权重总和校验写错而失败（未产出）；执行 002 为正式结果。

证据：`attempts/002/artifacts/train-data.yaml`、`report.json`。
