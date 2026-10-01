# 多说话人转写：qwen3-asr-sot-timestamps-2gpu-20260920

<!-- experiment-navigation -->

这是历史实验，产物保留原位。输入、完成条件、进展和结果见[任务导航](tasks/README.md)及原始方案；本次仅补齐导航，未重跑或重判质量。

执行状态：待核实；质量状态：沿用已有证据，缺少记录时保持未核实。

## 原始概览

# 时间戳 SOT 两卡试训

本轮使用通过自动对齐检查的 **1,644,206 条完整训练混音**，从上一阶段
`qwen3-asr-sot-v2-2gpu-20260916/training/checkpoint-60000` 初始化，计划两卡全参数训练
2,000 步。**两卡训练已完成 2,000 步**，500、1,000、2,000 步外部评测均已执行完成。
启动时两卡均通过第 5、20 步参数更新检查；见 `startup-verification.json`。
相同时间戳提示词下的起始模型评测已完成，496 条均未输出有效时间戳格式。

## 数据与指标

- 本地派生版本：`synthetic-v2-timed-trial-v1-20260920`。原始版本、混音、完整文字、
  来源身份和训练/开发/测试划分保留。355,794 条训练混音涉及待复核源标注，记录在
  `excluded-train.jsonl.gz`；不截断发言或伪造 clean 标记。
- 原始检查中有一条相邻边界相差一个浮点 ULP，被误判为重叠。修正比较后重新生成该
  shard；真实重叠仍拒绝。见 `repair_float_record.py` 与 `data-audit.json`。
- 固定开发集保持上一阶段全部 **496 个 ID 和原文字**。转写及说话人数指标全量计算；
  时间误差使用 465 条可用时间参考，覆盖率 **93.75%**，另 31 条继续参与转写评测。
  该时间指标不能外推为完整开发集时间精度。
- 原 62 个训练条件的采样权重保留，SOT / 中文 ASR / 英文 ASR 为 60/30/10，KL=2。
  时间戳样本不做变速或混响，防止音频时间与监督目标不一致。
- 时间目标为每句首个对齐字词开始至末个字词结束，不是逐字时间戳。自动通过不等于
  已人工确认边界精度。

冻结对齐索引：`../sot-timestamps-recovery-20260920/recovered/source-alignments.sqlite`，
SHA-256 为 `dd60f77acc2d164983d5906d2e8199f8f1edb7f65992768c1c89605c27567f5d`。
`fixed-dev.jsonl` 同时固定转写和可用时间参考；评测结果保存其 SHA-256 并严格比较协议。

## 训练与门槛

`launch.sh` 使用独立输出目录和冻结代码，创建新 AdamW 优化器及余弦计划；预热 100 步，
梯度累积 4，encoder / aligner / LLM 峰值学习率分别为 2e-5 / 2e-5 / 1e-5。
`training_audit.py` 检查起始权重、参数组、实际数据量，并在第 5、20 步检查三个模块
的真实有限参数更新。每 500 步保存模型，前 500 步及每 1,000 步执行外部评测。

普通 ASR teacher 与原基线不变，中文 CER 零退化门槛不变。
`baseline-asr/` 指向原始基线；`starting-asr/` 保存 60000 步起点的既有评测。
起点此前未通过全部中文保持检查，因此这次是探索试训，不能视为已通过准入的模型。
`baseline-sot/` 使用起始权重和新的时间戳提示词；各候选使用同一协议。
评测记录保持检查失败，训练可继续探索；不会把失败写成通过。

## 验证与进程记录

- `tests/test_sot_timestamps.py`：18 项通过。
- `evaluator-smoke.json`：确认没有时间参考的开发样本仍计入转写错误，时间统计明确分母。
- `asr-protocol-check.json`：512 条普通 ASR 开发样本指纹与原基线一致。
- `replay-cache-reuse.json`：中文回放复用上一阶段 14,567,995 条缓存。数据版本、筛选条件、
  源文件状态、采样率和增强参数完全一致；缓存身份仅多出两个该来源未使用的目录别名。
  普通 ASR 转换实现一致，未完成的重建文件另存保留；没有改动或减少回放数据。
- `cache.log`、`baseline-sot.log`、`training.log`：缓存、起始评测、训练日志。
- `training/audit-*.json`：每张卡的启动和参数更新检查。
- `training/retention-evaluations/`：普通 ASR 与固定 SOT 评测及比较结果。

## W&B 同步状态

用户已明确要求今后每次实验启动都同步拉起 W&B，已写入项目 `AGENTS.md` 和 SOT 启动说明。
本轮完整训练、性能及三次评测共 **407 个指标事件已上传并逐条核验**；远端训练步数为
2,000，评测步数为 500、1,000、2,000，run 状态为 `finished`。时间指标附带实际参考覆盖率。
已从用户配置的 `~/.bashrc` 加载凭据，未将凭据写入项目文件。

[W&B 运行页面](https://wandb.ai/1016097967-amphion/open-audio-llm/runs/qwen3-asr-sot-timestamps-2gpu-20260920)
；核验结果见 `wandb-sync/remote-verification.json`。

## 剩余标注问题

更换对齐器可能改善剩余边界，但没有证据保证清零；本轮不以增加对齐器作为开训前提。
Qwen 上游已有相同版本零时长问题的用户报告，见
[Qwen3-ASR issue #197](https://github.com/QwenLM/Qwen3-ASR/issues/197)。
独立多语种 CTC 对齐需要额外模型及中文转写预处理，参考
[PyTorch 多语种强制对齐教程](https://docs.pytorch.org/audio/main/tutorials/forced_alignment_for_multilingual_data_tutorial.html)。
这些资料支持另行评估补标方法，不构成本轮自动标注精度的证明。
