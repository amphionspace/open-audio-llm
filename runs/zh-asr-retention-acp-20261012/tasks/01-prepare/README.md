# 01-prepare：数据配置

输入：对话轮冻结的数据配置（`zh-dialogue-mix-acp-20261010` 01-prepare attempt 003），以及从共享登记目录复制的新来源条目（`shared/catalog-additions.jsonl`）。

完成条件：`train-data.yaml` 权重合计 100，对话轮来源合计 75、内部比例不变，新增单人普通格式来源合计 25；`report.json` 记录输入的 sha256。

结果：attempt 001 完成。新增 `wenetspeech_clean` 18、magicdata 4、aidatatang 1.5、primewords 1、aishell3 0.5；`format_policy.plain_asr_sources` 记录这 5 个来源。
