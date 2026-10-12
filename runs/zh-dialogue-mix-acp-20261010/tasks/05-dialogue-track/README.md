# 训练期间的对话评测跟踪

用户要求（2026-10-11）：不等训练结束，提前看新数据有没有被学进去。

输入：选模执行 003 复制完成的每个保存点（`models/checkpoint-<步数>/complete.json`，每 2000 步）；面板与结束评测相同的 AISHELL-5 dev、CS-Dialogue test 120 秒窗口；开发机 GPU 3，vLLM 服务。

完成条件：训练与选模都结束后，每个已保存的 checkpoint 在 `attempts/<编号>/artifacts/history.jsonl` 中有一行（cpER、去复读 cpER、复读条数、说话人数准确率、DER），并同步到 W&B（前缀 `dialogue`）。

对照：起点 14000 步 AISHELL-5 dev 36.86%（去复读 34.98%，2 条复读），CS-Dialogue test 22.09%（去复读 13.08%，7 条复读），见 [04-start-eval](../04-start-eval/README.md)。
