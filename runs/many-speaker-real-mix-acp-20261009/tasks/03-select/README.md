# 开发机异步选模与结束评测

输入：`02-train` 的 checkpoint（本地或对象存储）、起点同链路基线、选模面板与结束评测面板。

做法：见实验 README“选模与结束评测”。停止文件写入 `control/stop-request.json`。

完成条件：训练结束且全部 checkpoint 已评测；`final-results.json` 写出。执行完成与质量达标分开记录。

进展：尚未启动。

证据：`attempts/*/artifacts/baseline.json`、`selection-history.json`、`selection-decision.json`、`models/*/complete.json`（权重来源）、`final/`、`final-baseline/`、`final-results.json`。
