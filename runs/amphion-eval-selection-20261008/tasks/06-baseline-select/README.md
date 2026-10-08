# 06-baseline-select / 06-baseline-panel-moss：独立面板基线

- 06-baseline-select：`scripts/select_checkpoints.py` 不连接训练，只把起点 8000 步作为 step 0 走完“起服务（GPU 7）→ 面板与多人 70 条并行评测 → 关服务 → 记录”。结果 AliMeeting dev 22.12%、NOTSOFAR dev 125.77%、多人 70 条 112.51%，复读合计 17 条；起服务 3.5 分钟，评测 17 分钟。证据 `attempts/001/artifacts/baseline.json` 与 `checkpoints/step-0/*/attempts/001`。
- 06-baseline-panel-moss：MOSS AliMeeting dev 15.64%、NOTSOFAR dev 37.73%，无复读。
