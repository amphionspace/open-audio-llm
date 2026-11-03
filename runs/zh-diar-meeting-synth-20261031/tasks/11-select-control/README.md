# 对照训练选点

- 输入：[configs/select-control.yaml](../../configs/select-control.yaml)，基线 [shared/baseline-step0.json](../../shared/baseline-step0.json)（起点模型 3 次默认解码的均值，目标 0.20727）。
- 规则：`smoothing: 3` 的平滑配对规则，窗口 bootstrap，95% 置信区间上界 < 0 才算好于基线。评测模板沿用第二轮 I 组。
- 完成条件：所有 checkpoint 评完，W&B 远端核验。

## 进展

未启动，随对照训练提交后启动。
