# 02-export-zh245：现有 245 条中文选模集转为本地评测数据

- 输入：`unified-diarization-asr-20261001` prepare-full-training-data 执行 001 的 clips 与音频（AISHELL-4 / AliMeeting 官方 train 留出窗口，起点模型训练见过的会议）。
- 完成条件：AudioRecord 与音频索引的根目录别名直接指向原音频目录，不复制、不重编码。
- 结果：执行 001 因相对路径错误失败（保留）；执行 002 完成，aishell4 100 条、alimeeting 145 条。只用于一致性核对，不是共享登记。
