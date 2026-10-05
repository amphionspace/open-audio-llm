# Qwen3-ASR-1.7B 热词微调

原生 Qwen3-ASR 权重与 Catalog 数据在取样时解码、采样候选热词并增强。`amphion_asr_1.7b` 是插件注册名。TS 回放见 [TS_ASR.md](TS_ASR.md)，全说话人转写见 [SOT.md](SOT.md)。

已验证训练环境为 Python 3.11、torch 2.10.0+cu128、Transformers 4.57.6、ms-swift 4.1.0、qwen-asr 0.0.6、PEFT 0.18.1。安装 `data`、`swift`、`qwen3-asr` 可选依赖；原生卷积 hook 和并行编码依赖 qwen-asr 0.0.6 的内部接口，其他版本未作等价验证。

填写 [训练配置](../../configs/train/qwen3-asr.yaml) 的模型、Python 和设备，以及 [数据配置](../../configs/data/qwen3_asr_hotwords.yaml) 的 Catalog/roots：

```bash
open-audio-llm train --config examples/configs/train/qwen3-asr.yaml --dry-run
open-audio-llm train --config examples/configs/train/qwen3-asr.yaml
```

LoRA rank/alpha 为 32/64，冻结音频编码器和连接层；每卡最多 8 条，学习率 5e-5、cosine、5% warmup，500 步。每 250 步保存并计算 dev loss，保留 2 个 checkpoint；这些值均显式保存在 YAML，CLI 不接受额外业务参数。

热词放在 system 的 `Hotwords: Alice,Bob`；user 为单个 `<audio>`；assistant 为 `language English<asr_text>Hello Alice.`。普通 ASR 的 system 为空。数据版本和前 10000 条/来源的选择保持不变，不代表全量随机抽样。

checkpoint 和模型对比使用 vLLM。LoRA 先按 [merge-lora.yaml](../../configs/model/merge-lora.yaml) 合并，基础模型和候选分别按固定 [评测配置](../../configs/eval/comparison.yaml) 运行；保持样本、候选词、原始标签和解码条件一致。不得使用历史原生 PyTorch 评测入口作为自动回退。

需要把训练结果交给外部评测时，在训练 YAML 的 `task.arguments` 写 `--checkpoint_handoff: {path: "{attempt}/artifacts/checkpoint-handoff.json"}`。训练结束后由 rank 0 写入交付的 checkpoint、选择方式、全局步数、基础模型和 `tuner_type`。默认 `--checkpoint_selection: final`，要求末步已保存，不按目录名猜测最新 checkpoint；选 `best` 时使用 Trainer 记录的最佳 checkpoint。

训练与评测均由统一入口同步 W&B 并核验远端指标。性能字段见 [训练性能日志](../../../docs/training_performance.md)，记录规范见 [实验规范](../../../docs/experiments.md)。
