# 起点在新增结束评测集上的结果

输入：中文远场配比选出的 checkpoint-14000（本轮起点）；面板 AISHELL-5 dev 与 CS-Dialogue test 的 120 秒窗口，模板与选模结束评测相同（`configs/eval-final-*.yaml`），vLLM 服务。

完成条件：两个面板 summary 写入 `attempts/<编号>/artifacts/start-results.json`，W&B 有对应 run。

用途：选模结束评测只和链起点（8000 合并模型）对照；本任务给出本轮起点，用于直接判断 AISHELL-5 与 CS-Dialogue 训练数据的收益。
