# 03-select：开发机异步选模

每 1000 步一个保存点，在 GPU 6 上起 vLLM 服务，评测 AliMeeting/NOTSOFAR dev、6–12 人合成和单人 ASR dev（WenetSpeech、KeSpeech 各 1000 条，普通格式）。起点 18000 步作为 step 0 基线实测。

判定见实验 README。执行完成与质量达标分开记录。
