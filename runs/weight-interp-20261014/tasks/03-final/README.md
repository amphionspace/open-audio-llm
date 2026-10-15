# 03-final

输入：10% 插值模型 `tasks/01-merge/attempts/001/artifacts/base-10`；配置 `configs/eval-final.yaml`。

做法：部署推理栈（vLLM，GPU 3）跑结束面板 meeting-180s、CHiME-6、fixed338、AISHELL-5 dev、CS-Dialogue test，与 Emilia2 轮 6000 步在同模板同数据上的结束面板对照（不重测 6000 步）。

完成条件：五个面板都产出 summary.json，W&B 远端核验通过。

进展：2026-10-14 启动。
