# 开发机选模（GPU 6，端口 8851）

证据：`attempts/*/artifacts/selection-history.json`、`selection-decision.json`、`final/`、`final-results.json`。

执行 001 跟随训练执行 001，训练失败后自动退出；执行 002 跟随训练执行 002。

执行 003（2026-10-10 用户要求）：结束评测在 meeting180、CHiME-6 dev、固定 338 条之外新增 AISHELL-5 dev（`aishell5_dev`）与 CS-Dialogue test（`cs_dialogue_test`）的 120 秒窗口，读共享登记；这两个集合没有旧结果，结束时同样测链起点模型。执行 002 在任何评测前停止，改由执行 003 跟随训练执行 002。
