# 训练入口与复现

训练直接从 Catalog 在线构造数据，运行参数集中在 [YAML 配置](../../examples/configs/README.md)。先填写模型、Python、GPU 和数据路径，预览确认后启动：

```bash
open-audio-llm train --config examples/configs/train/sft.yaml --dry-run
open-audio-llm train --config examples/configs/train/sft.yaml
open-audio-llm train --config examples/configs/train/grpo.yaml
```

SFT 的冻结范围通过 `task.arguments` 中的 `--freeze_vit`、`--freeze_aligner`、`--freeze_llm` 配置。阶段切换先按 [模型配置](../../examples/configs/model/merge-lora.yaml) 合并 adapter，再将下一阶段模型路径写入配置。

GRPO 使用普通 Catalog 配方，不支持 SFT 的 samples/reps、动态 batch 或 SpecAugment；示例显式使用 vLLM server。先启动 [rollout 配置](../../examples/configs/rollout/vllm.yaml)，两者使用不同 GPU。验证全局 batch 必须能被 `--num_generations` 整除。

```bash
open-audio-llm rollout --config examples/configs/rollout/vllm.yaml
open-audio-llm model --config examples/configs/model/merge-lora.yaml
```

新执行自动分配 `attempts/编号`。恢复失败执行必须显式指定已有记录，且任务声明支持恢复；完成记录不会默认重跑。训练的模型/优化器恢复路径写在 YAML 的 `--resume_from_checkpoint` 中，不接受环境或 CLI 覆盖。只保存模型时，须作为新训练初始化，不能宣称恢复优化器。

每次训练和评测同步启动 CPU W&B 记录器。只有远端 run URL 和启动指标核验通过后，才启动任务；退出后再次核验上传指标和远端结束状态。默认 entity 为 `1016097967-amphion`，project 为 `open-audio-llm`。

历史 2026-09-11 GRPO 检查使用 Python 3.11.15、torch 2.10.0+cu128、Transformers 4.57.6、ms-swift 4.1.1、TRL 0.29.1、DeepSpeed 0.18.9、PEFT 0.18.1。单卡 ZeRO-2 LoRA 完成 2 步，loss 为 0.707788 / 0.710823；168 个 adapter 张量发生变化。历史本地生成验证不代表新的 vLLM rollout 已完成端到端验证，也不代表识别能力提升。

详细采样和恢复语义见 [在线训练数据](online-training-data.md)，实验组织见 [实验规范](experiments.md)。
