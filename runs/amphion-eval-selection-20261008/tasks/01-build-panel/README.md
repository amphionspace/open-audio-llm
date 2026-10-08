# 01-build-panel：构建独立选模面板

- 输入：AliMeeting Eval（`meeting-long-v1-20260920` dev_120s）、NOTSOFAR dev 与 CHiME-6 dev（`real-meetings-v1-20261008` dev_120s）的窗口记录和音频索引。
- 完成条件：全部窗口切出、远场通道取均值、写成 16 kHz PCM16，生成 AudioRecord、音频索引与 catalog 条目。
- 结果：执行 001 完成；alimeeting_dev 138 条 4.21 h、notsofar_dev 146 条 3.73 h（36 场各 1 台设备）、chime6_dev 150 条 4.46 h。
- 证据：`attempts/001/artifacts/report.json`、`catalog-entry.yaml`；数据在 `/workspace/data/datasets/meeting_selection_panel/`（未做对象存储备份）。
