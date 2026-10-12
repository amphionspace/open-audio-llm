# 开发机异步选模与结束评测

每个保存点在开发机 GPU 3、4 上起 vLLM 服务并用 AmphionEval 评测独立面板；停止规则与上一轮相同。训练期间不改写训练正在读取的数据目录。

证据：`attempts/*/artifacts/selection-history.json`、`selection-decision.json`、`final/`、`final-results.json`。
