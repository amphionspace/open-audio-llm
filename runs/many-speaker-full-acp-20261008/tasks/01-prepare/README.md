# 准备多人训练数据

输入：已清理的统一转写配方、固定版本单人合成源的原音频和对齐、AISHELL 官方 dev、AMI 官方 train。保留既有隔离记录和 clean 实际过滤。

完成条件：6–12 人每档 1000 条训练、10 条开发；原始说话人分区不重叠；完整时间戳目标；所有原来源保留；总配比 50/18/20/10/2；CPU 数据预检通过。

执行 001 因历史对齐 Catalog 没有 dev 分区失败，改用 AISHELL 官方 dev。执行 002 检出部分源说话人可用句子不足 180 秒混音，未发布训练数据；执行 003 加入独立语音储备检查后检出原机迁移路径缺失；执行 004 使用固定版本录音清单按 recording ID 解析现有原文件，并保留源路径、清单校验和与对齐依据。旧执行与产物保留。

证据：`attempts/004/artifacts/report.json`、`source-provenance.json`、`train-data.yaml`、`catalog.jsonl`、`selection-clips.jsonl`。数据通过自动检查不等于人工听审通过。
