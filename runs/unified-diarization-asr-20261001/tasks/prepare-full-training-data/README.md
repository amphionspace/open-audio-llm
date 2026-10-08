# 质检后的清理数据与选模集

排除 2,814 个窗口（`R1021_M1947` 80 个、空目标 921 个、留出会议 1,813 个）；去掉 12 个 dense 合成档并按比例放大其余多人合成权重（保持 20/40/40）；合成与单人 ASR 以 0.5 概率加 MUSAN 噪声（SNR 0–15 dB），真实会议不加。选模集：AISHELL-4 留出 10 场各 10 段约 30 秒，AliMeeting 留出 8 场全部 120 秒段，共 245 段 4.9 小时。

证据：`attempts/001/artifacts/report.json`、`excluded-records.txt`、`train-data.yaml`、`selection/`。
