# 实验与分析记录：catalog-mux-gpu23-20260909

<!-- experiment-navigation -->

这是历史实验，产物保留原位。输入、完成条件、进展和结果见[任务导航](tasks/README.md)及原始方案；本次仅补齐导航，未重跑或重判质量。

执行状态：待核实；质量状态：沿用已有证据，缺少记录时保持未核实。

## 原始概览

# YAML / Catalog 混合 SFT 验证

结论：双卡连续训练 8 步与从 checkpoint-3 恢复到第 8 步，后续逐步 loss、
最终 340 个 adapter 张量和 Catalog sampler 状态完全一致。
结果见 `summary.json`，复核命令为 `python summarize.py`。

- GPU：物理 2、3；实验结束后已释放。
- 环境：`/ai_sds_wuzz/MODELS/miniconda3/envs/amphionft`。
- 模型：沿用 `../catalog-gpu23-20260909/initial-model`，Qwen3-ASR-0.6B encoder +
  Qwen2.5-0.5B-Instruct + 初始随机 MLP connector。
- 训练：LoRA rank 8 / alpha 16，bf16，DeepSpeed zero2，gradient accumulation 2，
  每 rank 2 个 persistent workers，seed 42。
- 配方：`reference.yaml`；librispeech dev 每轮 8 条 × 1，aishell dev 每轮 4 条 × 2，
  两来源开启逻辑分片轮换；22 秒 / 4 条的 batch 上限、2 个时长桶。
- 在线增强：speed + SpecAugment。
- 连续训练：`reference.sh`，8 updates，4 个数据 epoch。
- 恢复训练：`resume.sh`，从实际数据 epoch 1 已消费 2 / 3 个本地 batch 的位置恢复。
- 最终 loss：两者均为 3.777583122253418；adapter 最大绝对差为 0。

最初的完整链路检查使用 `run.sh`，产物位于 `full/`。
`reference.sh` 调整了保存间隔，以便验证 epoch 中途的恢复。

仅作数据与训练链路验证：开发集被用作小规模训练来源，没有进行泛化或 WER/CER 评估。
当前 HF 日志中的 epoch 是训练循环推算值，恢复后可能偏移；实际的数据 epoch
以 checkpoint 的 `catalog_sampler.json` 为准。动态 batch 下 Trainer 的固定条数
samples/s 估计不能用作真实音频吞吐量。
