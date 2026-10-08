# 开发机异步选模与结束评测

输入：`02-train` 执行 003 的 checkpoint-500 及其任务内评测结果，执行 004 后续保存的 checkpoint。

做法：在开发机上轮询完整保存的 checkpoint（以 `catalog_sampler.json` 写出为准），逐个运行与原任务内相同的中文 245 条与多人 70 条选模评测，脚本、vLLM 0.18.0 解码与打分口径不变。中文连续两次比起点差超过 1 个百分点时，向训练执行写 `stop-request.json`，训练回调每 5 步由 rank 0 读取并广播停止；停止会比任务内评测晚一个评测间隔左右。训练完成后，对选出的 checkpoint 运行固定 338 条中文集和 180 秒会议集。

原因：任务内评测期间 16 张卡全部暂停、只有 1 张卡推理；第 500 步一次评测（含起点多人基线）超过 10 分钟。

完成条件：训练执行结束且全部 checkpoint 已评测；`final-results.json` 写出。执行完成与质量达标分开记录。

证据：`attempts/*/artifacts/selection-history.json`、`selection-decision.json`、`checkpoints/`、`final-evaluation/`、`final-results.json`。
