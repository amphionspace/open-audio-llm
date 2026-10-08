# 统一 Diarization 格式训练

## 当前状态（2026-10-08）

当前最好的点仍是 LoRA 002 / checkpoint-8000（A800 固定集 cpCER 20.44%）；对标 MOSS-Transcribe-Diarize（官方 vLLM）15.53%。10-06 至 10-08 在本机 8×A800 上的续训、SFT 和全参训练都没有超过 8000；全参训练于 10-08 暂停，后续训练改用提交方式，不再使用本机算力。

- 阶段结论与执行表：[docs/experiments/2026-10-08](../../docs/experiments/2026-10-08/README.md)
- 标准评测：固定集 [evaluate-a800](tasks/evaluate-a800/README.md)、180 秒会议评测集 [build-meeting-benchmark](tasks/build-meeting-benchmark/README.md) → [score-meeting-benchmark](tasks/score-meeting-benchmark/README.md)；MOSS 基线输出已缓存
- 已确认短板：6 人以上、英文会议、180 秒片段复读循环
- 留存：执行记录、评测产物和保留的 checkpoint 在 `whai:open-audio-llm/runs/unified-diarization-asr-20261001/`，与本地路径一致

## 早期记录（原机 2×RTX 5090，2026-10-01 至 10-05）

2026-10-04 已在本机两张 RTX 5090 32GB 上启动[新的 LoRA 持续训练](tasks/local-training/README.md)。首个执行因调度器实际忽略 `timescale=10000` 已停止并保留；第二个执行将从 checkpoint-1000 重新开始，使用修复后的调度器。选择已有同机测评的 checkpoint-1000 为基座，使用修复配比后的全量数据；不继承退化 checkpoint-71000 或错误调度执行的优化器状态。

当前配比为单说话人回放 20%、多人合成 40%、真实会议 40%，保留固定版本的 4,023,765 条有效训练记录及既有过滤规则；合成单人样本计入回放。小来源循环读取，完整覆盖轮按最慢来源计算，无固定子集或业务训练上限。新的本机配置明确采用监督回放，关闭无效 KL；500 步 warmup 后 inverse-sqrt 衰减，冻结基座，仅训练语言模型 LoRA。已接入固定集周期 vLLM 评测，首次第 500 步，之后每 2000 步；当前启动正确不代表质量达标。

历史 001 使用的[自然比例数据配置](shared/data/full-train.yaml)、[训练方案](shared/full-training-plan.json)和[整轮核验](shared/full-coverage-verification.json)保留供追溯；其 125,981 步/epoch 及末批补齐 4 条不适用于当前配比。

单说话人 ASR 使用有说话人标注的 AISHELL、KeSpeech、CommonVoice EN clean 和 LibriSpeech，转换为 `[S1] text`；缺少可靠说话人监督的 WenetSpeech 保持隔离。固定 ASR 回访数据保留在 `evaluation` 分组。

进展：2026-10-04 已确认训练停止，日志最后为 71,225 步，最新完整 checkpoint 为 71,000 步，约 0.556 数据 epoch。全量预检、AdamW 状态恢复及两张 GPU 的权重更新审计此前通过。旧执行状态文件未收尾，不能据它宣称训练仍在运行。每 500 步保存 checkpoint，保留最近 3 个，完整数据 epoch 约需 125,981 个 optimizer step。

评测：338 条固定会议样本、28 场会议，两个数据集等权平均 cpCER：初始 22.28%、旧混合 20.67%、统一格式第 1000 步 20.94%；5 秒容差 tcpCER：23.18%、21.05%、21.26%。相对初始有收益，相对旧混合无显著收益；格式合规率下降，不能单独归因于数据清洗。48 条 ASR 回访样本仅作小规模回归检查。

最新测评：第 71,000 步与第 1,000 步都在当前 RTX 5090、vLLM 内置 Triton attention 上重新生成。平均 cpCER 从 21.43% 升至 29.34%，达到 token 上限的样本从 2 条增至 25 条；格式合规率提高，但会议能力明显退化。本次 338 条输出全部进入主 cpCER，不能将 loss 上行只当作正常噪声。

证据见 [全量训练](tasks/full-training/README.md)、[最新测评报告](tasks/evaluate-71000/attempts/002/artifacts/evaluation/report.md)、[完整结果](tasks/evaluate-71000/attempts/002/artifacts/evaluation/summary.json)和 [评测 W&B](https://wandb.ai/1016097967-amphion/open-audio-llm/runs/unified-diarization-asr-71000-eval-002)。评测执行完成，训练保持停止。

当前任务入口：`open-audio-llm train --config runs/unified-diarization-asr-20261001/configs/train-local.yaml`。历史 full-training 的停止状态及评测结果保留供追溯。

<!-- execution-status -->

执行状态：partial；质量状态：未核实。

开始：未知；结束：未结束；退出码：未知。

<!-- /execution-status -->
