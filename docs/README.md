# 文档

按用途分为五类：**上手**、**指南**（怎么做）、**参考**（是什么 / 为什么）、**模型**（已产出的权重）和 **实验 / 归档**（只读历史记录）。

## 上手

| 文档 | 内容 |
| --- | --- |
| [安装](get-started/installation.md) | 训练、vLLM 推理与 serving 环境的安装和校验 |

## 指南

| 文档 | 内容 |
| --- | --- |
| [训练入口与复现](guides/training.md) | SFT / GRPO / rollout 的配置化启动、冻结范围与恢复 |
| [在线训练数据](guides/online-training-data.md) | Catalog 在线取样、clean 过滤、增强和断点恢复 |
| [ACP 多机训练](guides/acp.md) | 通过 `cluster` 段向 SenseCore ACP 算力池提交任务 |
| [多目标说话人 SOT](guides/multi-target-sot.md) | 0～3 段注册音频的日志转写训练与评测 |
| [使用 AmphionEval 评测](guides/evaluation.md) | 训练与评测的边界协议、环境和评测步骤 |
| [部署](guides/deployment.md) | vLLM 服务、TS-ASR、Compose、自包含镜像与 Kubernetes |
| [实验目录与配置规范](guides/experiments.md) | 迭代流程、实验目录、执行记录、W&B 与对象存储 |
| [训练性能日志](guides/training-performance.md) | 吞吐、阶段耗时与 GPU 采样字段 |

## 参考

| 文档 | 内容 |
| --- | --- |
| [架构](reference/architecture.md) | 组件契约和模型组合 |
| [数据边界](reference/data-boundary.md) | 离线样本事实与在线训练随机性的边界 |
| [兼容矩阵](reference/compatibility-matrix.md) | 默认支持和可选 / legacy 路径 |
| [vLLM Triton Embedding Bypass](reference/vllm-triton-bypass.md) | vLLM 直接消费外部 audio embedding |
| [Legacy 依赖](reference/legacy-deps.md) | k2 / Zipformer 的处理方式 |
| [当前限制](reference/limitations.md) | 训练数据链路尚未支持的能力 |

## 模型

| 文档 | 内容 |
| --- | --- |
| [Qwen3-ASR-1.7B 三阶段 SFT](models/qwen3-asr-three-stage-sft.md) | 从底座到 checkpoint-34479 的数据、学习率和冻结配置 |
| [checkpoint-34479 评测结果](models/ckpt34479-evaluation.md) | ASR、热词、TS-ASR 和警务指标 |

## 实验归档

| 文档 | 内容 |
| --- | --- |
| [2026-10-06 至 10-08](experiments/2026-10-08/README.md) | 8×A800 续训、数据质检、180 秒会议评测集，与 MOSS 同口径对比 |
| [截至 2026-10-06](experiments/2026-10-06/README.md) | 训练轨迹、评测结果与权重下载 |
| [2026-09-22 至 09-29](experiments/2026-09-22-to-29.md) | SOT 会议训练阶段记录 |
| [历史记录](archive/README.md) | 早期上下文、旧计划与迁移决策 |
