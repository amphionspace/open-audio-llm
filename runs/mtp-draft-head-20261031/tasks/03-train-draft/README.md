# 03 正式草稿头训练（已备好，未启动）

- 设计：EAGLE-3 草稿头，融合骨干第 2 / 14 / 25 层，3 步展开，目标是对冻结模型下下个 token 分布的 KL，训练文本用数据标注（依据见 02）。
- 输入：**GRPO 之后的权重**（冻结）、`configs/train-data.yaml`（054 训练配方原样）、本分支代码快照 `shared/code`。
- 配置：`configs/train-draft.yaml`，提交前只需把 `--model` 换成新权重。ACP 1 节点 8 卡、全局 batch 192（12 × 8 × 累积 2）、20000 步（384 万条）、学习率 1e-4 warmup 300 余弦到 1e-5、不用 DeepSpeed。按独占时的 0.48 s/步估计约 2.7–3.5 小时。
- 完成条件：训练正常结束；用 `checkpoint-20000/eagle/` 跑 02 的完整协议（`configs/spec-check-final.yaml` 等，把草稿路径换成新 checkpoint），并确认接受长度不低于 10000 步草稿头。384 万条这个预算本身没有实测，接受率随预算的增长是否继续要在这一步确认。
- 进展：未启动，由用户作为记录实验提交。`--dry-run` 已核对：8 卡 torchrun、`--draft_head_arch eagle3`、ACP 任务名 `AmphionASR-mtp-draft-eagle3-3step-20k-1node-001`。

## 提交

```bash
cd /workspace/workspace/mingdong/repo/oal-mtp
# 1. 代码快照：shared/code 已按 3df5179 生成（任务容器只挂载 /workspace，训练只读快照；旧快照移到 shared/code-e9c9118）。
#    之后若改了代码，先移走旧快照再重新生成：bash runs/mtp-draft-head-20261031/scripts/snapshot_code.sh
# 2. 预览（不启动进程、不写外部系统）
env -u PYTHONPATH /workspace/workspace/mingdong/miniconda3/envs/oal-train/bin/open-audio-llm train \
    --config runs/mtp-draft-head-20261031/configs/train-draft.yaml --dry-run
# 3. 提交 ACP（任务名自动带 AmphionASR- 前缀，W&B 名 AmphionASR-mtp-draft-eagle3-3step-20k-1node）
env -u PYTHONPATH /workspace/workspace/mingdong/miniconda3/envs/oal-train/bin/open-audio-llm train \
    --config runs/mtp-draft-head-20261031/configs/train-draft.yaml
```

启动器用环境里已安装的 `open-audio-llm`；训练进程只读配置里的 `runtime.pythonpath`（`shared/code/src`），所以提交时去掉继承的 `PYTHONPATH`（否则任务名会被旧快照的前缀规则改写，见 2026-10-26 的 050）。

## 产物与使用

- 每个 checkpoint：`model.safetensors`（只含草稿头 `mtp_draft.*`，fp32，约 235 MB，续训用）、`optimizer.pt`、`eagle/`（vLLM 草稿目录：Llama 配置 + `fc`/`layers.0.*` bf16 权重 + `draft_info.json` 记录骨干路径与步数）。
- 上线：交付包 runtime 需要 `shared/pack-eagle` 里目标模型类的两处改动（`config.image_token_index` 一行、基类 `SupportsEagle3`，已合入 `src/open_audio_llm/tsasr/vllm_backend.py`），服务参数加
  `--speculative-config '{"method": "eagle3", "model": "<checkpoint>/eagle", "num_speculative_tokens": 3}'`。k=3 由 02 的 015–018 测速选定：各档不差于 k=2，48 路也没有变慢。

## 骨干更换

草稿头读的是骨干第 2 / 14 / 25 层的隐状态和骨干的词表投影，换骨干（例如 GRPO 之后的权重）必须把 `--model` 换成新权重重训；旧草稿头接在新骨干上仍然无损（贪心校验），但接受率会下降，没有测过。

<!-- execution-status -->

执行状态：未启动。

<!-- /execution-status -->
