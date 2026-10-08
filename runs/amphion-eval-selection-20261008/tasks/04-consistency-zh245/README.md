# 04-consistency-zh245：245 条中文集新链路评测

- 配置：`configs/eval-step8000-zh245.yaml`（`ae open-audio-llm meeting`，服务 03-serve）。
- 结果：执行 001 aishell4 10.63%、alimeeting 19.97%，宏平均 15.30%，复读 1 条；执行 002 为同配置重复运行，宏平均 15.44%（用于估计新链路自身波动）。W&B 均已核验 finished。
- 与旧脚本的逐条比较见 08-compare-consistency。
