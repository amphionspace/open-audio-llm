# 实验与分析记录：catalog-gpu23-20260909

<!-- experiment-navigation -->

这是历史实验，产物保留原位。输入、完成条件、进展和结果见[任务导航](tasks/README.md)及原始方案；本次仅补齐导航，未重跑或重判质量。

执行状态：待核实；质量状态：沿用已有证据，缺少记录时保持未核实。

## 原始概览

# GPU 2、3 在线数据训练实验

两组均完成 8 个更新步骤、2 个 epoch。使用相同初始权重、16 条训练样本和 4 条固定验证样本。

| 组别 | 最后一步训练 loss | 最终验证 loss | Trainer 耗时（秒） |
|---|---:|---:|---:|
| baseline | 3.6039 | 4.1492 | 17.64 |
| augmented | 3.5958 | 4.1411 | 11.74 |

增强组启用在线变速（0.9/1.0/1.1）和 SpecAugment，概率均为 1。
两组使用 2 个 DataLoader workers/rank，persistent_workers=true，bf16、LoRA rank 8、DeepSpeed ZeRO-2。
音频和标注直接由本地 audio-data-contract Catalog 读取，训练输入在取样时构造，无离线 ShareGPT 或增强音频文件。

模型的编码器和文本模型使用本地预训练权重，MLP 连接层随机初始化。样本很少，结果仅验证训练管线，不证明 ASR 精度改善；未计算 WER/CER。
耗时是 Trainer 记录，包含首次加载/预热与验证，不包含环境导入、模型初始化和数据清单加载时间。

复现：`bash run.sh`；准备权重：`prepare.py`；配置：`baseline.json`、`augmented.json`；完整指标：`summary.json`。
