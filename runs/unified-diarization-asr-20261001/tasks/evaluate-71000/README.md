# 第 71,000 步固定集测评

输入：[评测配置](../../configs/evaluate-71000.yaml)、全量训练停止前最新完整 checkpoint-71000、固定 338 条会议样本，以及 AISHELL、KeSpeech、LibriSpeech 各 16 条 ASR 开发样本。

完成条件：实际 vLLM 生成全部输出，计算 cpCER、两种容差 tcpCER、格式合规率、说话人数准确率和 ASR CER/WER，核验 W&B 远端指标与结束状态。执行完成和模型质量分别记录。

进展：checkpoint 的两片权重、708 个张量索引与推理配置已核验；训练进程已消失，日志停在 71,225 步。旧训练状态文件未收尾，不能据此宣称训练仍在运行。[002 执行](attempts/002/README.md)已完成，退出码 0，W&B finished 状态和 404 项实际指标已核验。

对照：第 1,000 步模型与最新模型在当前 RTX 5090 上重新推理，都使用 vLLM 内置 Triton encoder/decoder attention。初始模型和旧混合方案复用已保存的同协议预测；它们运行时 GPU/内核不同，主要结论以本次两模型的直接配对为准。新结果与原结果分别保存，不覆盖历史。

首次启动在 vLLM 自带 FlashAttention 2 内核处出现 PTX 工具链不兼容，失败记录保存于 attempts/001；重试显式选择 vLLM 内置 Triton attention，未回退到 Transformers 模型推理。

局限：会议集是已使用过的固定开发集；ASR 样本较少，仅作回归检查。按会议配对统计不作跨多次 checkpoint 测试校正。结果、runtime、日志和 W&B 证据见 attempts 中的逐次执行记录。

结果：同机重新推理的第 1,000 步至第 71,000 步，平均 cpCER 从 21.43% 升至 29.34%，增加 7.91 个百分点，按会议配对的 95% 区间为 +4.95 至 +10.46 个百分点。截断从 2 条增至 25 条。格式合规率提高，但实际会议能力退化，不能将执行完成视为质量达标。

逐语料、ASR 回归、配对时间指标及局限见[测评报告](attempts/002/artifacts/evaluation/report.md)与[完整结果](attempts/002/artifacts/evaluation/summary.json)。
