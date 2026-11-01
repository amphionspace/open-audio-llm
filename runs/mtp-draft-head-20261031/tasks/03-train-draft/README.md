# 03 正式草稿头训练（已备好，未启动）

- 输入：10.28 版本权重（冻结）、`configs/train-data.yaml`（054 训练配方原样）、本分支代码快照 `shared/code`。
- 配置：`configs/train-draft.yaml`。与 01/002 只差规模：ACP 1 节点 8 卡、全局 batch 192（12 × 8 × 累积 2）、20000 步（384 万条，约数据配方两轮）、warmup 300；学习率 1e-4 余弦到 1e-5、草稿步数 3、逐步衰减 0.8、不用 DeepSpeed 不变。按 002 的 0.46 s/步（每卡吞吐相同）估计 2.6–3.5 小时。
- 完成条件：训练正常结束；最后的 `checkpoint-20000/eagle/` 用 02 的 `configs/spec-check.yaml`（把草稿路径换成新 checkpoint）复测接受长度与速度，并与 01/002 的 2500 步草稿对比。
- 进展：未启动，由用户作为记录实验提交。

## 提交

```bash
cd /workspace/workspace/mingdong/repo/oal-mtp
# 1. 冻结代码（任务容器只挂载 /workspace；已存在则不覆盖）
bash runs/mtp-draft-head-20261031/scripts/snapshot_code.sh
# 2. 预览（不启动进程、不写外部系统）
env -u PYTHONPATH /workspace/workspace/mingdong/miniconda3/envs/oal-train/bin/open-audio-llm train \
    --config runs/mtp-draft-head-20261031/configs/train-draft.yaml --dry-run
# 3. 提交 ACP（任务名自动带 AmphionASR- 前缀，W&B 名 AmphionASR-mtp-draft-eagle-k3-on-1028-054avg3-20k-1node）
env -u PYTHONPATH /workspace/workspace/mingdong/miniconda3/envs/oal-train/bin/open-audio-llm train \
    --config runs/mtp-draft-head-20261031/configs/train-draft.yaml
```

启动器用环境里已安装的 `open-audio-llm`；训练进程只读配置里的 `runtime.pythonpath`（`shared/code/src`），所以提交时去掉继承的 `PYTHONPATH`（否则任务名会被旧快照的前缀规则改写，见 2026-10-26 的 050）。

## 产物与使用

- 每个 checkpoint：`model.safetensors`（只含草稿头 `mtp_draft.*`，fp32，约 235 MB，续训用）、`optimizer.pt`、`eagle/`（vLLM 草稿目录：Llama 配置 + `fc`/`layers.0.*` bf16 权重 + `draft_info.json` 记录骨干路径与步数）。
- 上线：交付包 runtime 需要 `shared/pack-eagle` 那一行 `config.image_token_index`（已合入 `src/open_audio_llm/tsasr/vllm_backend.py`），服务参数加
  `--speculative-config '{"method": "eagle", "model": "<checkpoint>/eagle", "num_speculative_tokens": 2}'`。

## 骨干更换

草稿头读的是骨干最后一层（final norm 之后）的隐状态和骨干的词表投影，换骨干（例如 GRPO 之后的权重）必须把 `--model` 换成新权重重训；旧草稿头接在新骨干上仍然无损（贪心校验），但接受率会下降，没有测过。

<!-- execution-status -->

执行状态：未启动。

<!-- /execution-status -->
