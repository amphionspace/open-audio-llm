# 真实会议窗口与原始标注核对

逐窗口（39,407 个）对照 AISHELL-4 / AliMeeting 原始人工标注：窗口内无漏标语音，句子时间与文本一致，无越界时间戳。发现 0 说话人窗口（目标为空）和少量无效短窗口，已在 prepare-full-training-data 中移除。

证据：`attempts/001/artifacts/summary.json`、`examples.json`。
