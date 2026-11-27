# 02 挖题（K0 起点，第十轮的五份计划）

- **输入**：K0 checkpoint-3000，`configs/train-data-grpo.yaml`、`configs/mining/`（`scripts/make_configs.py grpo START`，从 kgrpo 的配置改）。
- **做法**：`scripts/mining.sh`，同 kgrpo，去掉两路 ctx 挖题，切计划时不给 context targets（不产生 ctx 计划）。
- **完成条件**：hotword、nonspeech、miss、ts-absent、ts-present 五份计划都非空，`summary.json` 写出。
- **进展**：未开始。
