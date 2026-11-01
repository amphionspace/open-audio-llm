# 01 草稿头短训验证

- 输入：10.28 版本权重 `/workspace/data/packages/amphionasr-1.7b-v3-tsrobust-054avg3/model`（全部冻结）；数据 `configs/train-data.yaml`（054 训练配方原样，只换缓存目录、去掉 validation）；代码 `feat/mtp-draft-head`（`--draft_head_steps 3 --draft_head_step_decay 0.8`）。
- 方法：ACP 开发节点 GPU 4–7（GPU 0–3 属于 GRPO 会话），`scripts/train_draft_acp.sh`；用户要求短训不接 W&B（`--report_to none`），正式训练走 `configs/train-draft.yaml`。
- 完成条件：训练跑通、checkpoint 里有可直接给 vLLM 的 `eagle/` 草稿目录，草稿头在 vLLM 里的接受率可测（见 02）。
- 进展：完成。

## 执行

| 执行 | 内容 | 代码 | 结果 |
|---|---|---|---|
| 001 | 冒烟 30 步，草稿头从骨干最后一层复制初始化 | e2cc7b6 | 跑通；第 1 步 KL 194 nat（最后一层 MLP 输出没有 final norm，logits 极端），改为 transformers 默认初始化（1d6d2f7） |
| 002 | 2500 步，4 卡 × 每卡 12 条 × 累积 2 = 96 条/步，峰值学习率 1e-4（warmup 100，余弦到 1e-5），每 500 步存 | 1d6d2f7 | 19 分钟，0.46 s/步，取数据等待 < 1%，显存峰值 10.8 GiB/卡 |

## 结果（002，训练数据上 teacher forcing，最后 5 次日志平均）

“第 k 步一致率”= 草稿第 k 步的 argmax 与冻结模型 argmax 相同的比例（不以前几步正确为条件）；“可接受草稿数”= 3 步提案中从第 1 步起连续一致的步数均值。

| 步数 | KL 第 1/2/3 步 | 一致率 第 1/2/3 步 | 可接受草稿数（满分 3） |
|---:|---|---|---:|
| 500 | 4.46 / 6.11 / 6.77 | 0.377 / 0.217 / 0.140 | 0.51 |
| 1000 | 2.97 / 4.71 / 5.41 | 0.504 / 0.284 / 0.207 | 0.74 |
| 1500 | 2.38 / 3.77 / 4.44 | 0.572 / 0.366 / 0.276 | 0.91 |
| 2000 | 2.23 / 3.52 / 4.15 | 0.577 / 0.389 / 0.301 | 0.95 |
| 2500（均值） | 2.18 / 3.28 / 3.82 | 0.598 / 0.430 / 0.348 | 1.04 |

学习率衰减到 1e-5 时曲线仍在下降，只训了 24 万条样本（数据配方一轮 200 万条），正式训练加大规模（03）。

## 证据

- `attempts/002/artifacts/training/checkpoint-2500/eagle/`：vLLM 草稿目录（`config.json` + `model.safetensors`，只含 `fc` 与 `layers.0.*`，bf16，117 MB）；`model.safetensors`（`mtp_draft.*`，fp32，续训用）。
- `attempts/002/artifacts/train-log.json`：每 10 步日志；`attempts/00N/logs/runner.log`：完整输出；`artifacts/training/code-commit.txt`、`code-diff.patch`：代码版本。

<!-- execution-status -->

执行状态：completed；质量状态：未核实。

<!-- /execution-status -->
