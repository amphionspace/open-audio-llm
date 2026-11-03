# 01 草稿头短训验证

- 输入：10.28 版本权重 `/workspace/data/packages/amphionasr-1.7b-v3-tsrobust-054avg3/model`（全部冻结）；数据 `configs/train-data.yaml`（054 训练配方原样，只换缓存目录、去掉 validation）；代码 `feat/mtp-draft-head`（`--draft_head_steps 3 --draft_head_step_decay 0.8`）。
- 方法：ACP 开发节点 GPU 4–7（GPU 0–3 属于 GRPO 会话），`scripts/train_draft_acp.sh`；001–004 不接 W&B，002 的日志事后补录到 [2t3givov](https://wandb.ai/1016097967-amphion/open-audio-llm/runs/2t3givov)；005 起经开发节点转发器在线记录。正式训练走 `configs/train-draft.yaml`。
- 完成条件：训练跑通、checkpoint 里有可直接给 vLLM 的 `eagle/` 草稿目录，草稿头在 vLLM 里的接受率可测（见 02）。
- 进展：完成。

## 执行

| 执行 | 内容 | 代码 | 结果 |
|---|---|---|---|
| 001 | 冒烟 30 步，草稿头从骨干最后一层复制初始化 | e2cc7b6 | 跑通；第 1 步 KL 194 nat（最后一层 MLP 输出没有 final norm，logits 极端），改为 transformers 默认初始化（1d6d2f7） |
| 002 | 2500 步，4 卡 × 每卡 12 条 × 累积 2 = 96 条/步，峰值学习率 1e-4（warmup 100，余弦到 1e-5），每 500 步存 | 1d6d2f7 | 19 分钟，0.46 s/步，取数据等待 < 1%，显存峰值 10.8 GiB/卡 |
| 003 | EAGLE-3 冒烟 30 步（`--draft_head_arch eagle3`，骨干第 2/14/25 层） | 46ff7b5 | 跑通，初始 KL 12.5（接近均匀分布，归一化层让起点好于 EAGLE-1 的 27） |
| 004 | 自蒸馏计时冒烟 20 步（EAGLE-1，`--draft_head_targets greedy`，每行都从提示生成，上限 256） | 96a80b8 | 34 s/步，是参考文本训练 0.46 s/步的约 73 倍；27% 的行贪心输出与标注不同。第一次尝试因特殊 token 取错失败，其产物被我用 `rm` 删掉（违反保留失败产物的规定），控制台输出保留在 003 的 `runner.log` 末尾 |
| 005 | EAGLE-3 2500 步，预算与 002 完全相同（4 卡、96 条/步、同学习率与调度），W&B [ygzvh5pw](https://wandb.ai/1016097967-amphion/open-audio-llm/runs/ygzvh5pw) | 96a80b8 + `code-diff.patch` | 0.48 s/步；结果见下 |
| 006 / 007 | EAGLE-1 / EAGLE-3 各 5000 步（48 万条），其余同 002 / 005 | dc8c77f | 0.46 / 0.48 s/步；留出请求接受长度见 02 的 012 |
| 008 / 009 | 自蒸馏 A/B：从 007 的 5000 步 EAGLE-3 草稿头出发各微调 500 步（同一批新样本，学习率 3e-5 余弦到 1e-5，`configs/train-data-ft.yaml`）；008 用标注文本，009 用冻结模型的贪心输出 | dc271b2 | 008 约 0.48 s/步；009 约 8.3 s/步（约 17 倍，取共享 GPU 之前的读数）；26% 的行贪心输出与标注不同，0.1% 触到长度上限。接受长度相同，见 02 的 013 |
| 010 / 011 | EAGLE-1 / EAGLE-3 各 10000 步（96 万条），W&B [esm5f1fj](https://wandb.ai/1016097967-amphion/open-audio-llm/runs/esm5f1fj) / [q4qljjfo](https://wandb.ai/1016097967-amphion/open-audio-llm/runs/q4qljjfo) | dc271b2 | GPU 与 GRPO 共享，1.09 / 1.25 s/步（各约 3 小时、3.5 小时，不代表独占速度）；训练集一致率与接受率见下 |

## 结果（002，训练数据上 teacher forcing，最后 5 次日志平均）

“第 k 步一致率”= 草稿第 k 步的 argmax 与冻结模型 argmax 相同的比例（不以前几步正确为条件）；“可接受草稿数”= 3 步提案中从第 1 步起连续一致的步数均值。

| 步数 | KL 第 1/2/3 步 | 一致率 第 1/2/3 步 | 可接受草稿数（满分 3） |
|---:|---|---|---:|
| 500 | 4.46 / 6.11 / 6.77 | 0.377 / 0.217 / 0.140 | 0.51 |
| 1000 | 2.97 / 4.71 / 5.41 | 0.504 / 0.284 / 0.207 | 0.74 |
| 1500 | 2.38 / 3.77 / 4.44 | 0.572 / 0.366 / 0.276 | 0.91 |
| 2000 | 2.23 / 3.52 / 4.15 | 0.577 / 0.389 / 0.301 | 0.95 |
| 2500（均值） | 2.18 / 3.28 / 3.82 | 0.598 / 0.430 / 0.348 | 1.04 |

同样口径的 EAGLE-3（005，2500 步均值）：KL 2.54 / 4.26 / 5.01，一致率 0.576 / 0.340 / 0.230，可接受草稿数 0.84，低于 EAGLE-1 的 1.04。

| 训练预算 | 结构 | KL 第 1/2/3 步 | 一致率 第 1/2/3 步 | 可接受草稿数 |
|---|---|---|---|---:|
| 2500 步 | EAGLE-1 | 2.18 / 3.28 / 3.82 | 0.598 / 0.430 / 0.348 | 1.04 |
| 2500 步 | EAGLE-3 | 2.54 / 4.26 / 5.01 | 0.576 / 0.340 / 0.230 | 0.84 |
| 5000 步 | EAGLE-1 | 1.21 / 1.59 / 1.86 | 0.747 / 0.680 / 0.634 | 1.70 |
| 5000 步 | EAGLE-3 | 1.29 / 1.85 / 2.25 | 0.776 / 0.685 / 0.593 | 1.69 |
| 10000 步 | EAGLE-1 | 0.83 / 0.98 / 1.10 | 0.810 / 0.783 / 0.760 | 2.03 |
| 10000 步 | EAGLE-3 | 0.70 / 0.85 / 0.89 | 0.859 / 0.832 / 0.805 | 2.22 |

草稿头规模：EAGLE-1 5870 万参数、112 MiB（bf16）；EAGLE-3 7130 万参数、136 MiB。

002 的学习率衰减到 1e-5 时曲线仍在下降，只训了 24 万条样本（数据配方一轮 200 万条），正式训练加大规模（03）。

## 证据

- `attempts/002/artifacts/training/checkpoint-2500/eagle/`：vLLM 草稿目录（`config.json` + `model.safetensors`，只含 `fc` 与 `layers.0.*`，bf16，117 MB）；`model.safetensors`（`mtp_draft.*`，fp32，续训用）。
- `attempts/002/artifacts/train-log.json`：每 10 步日志；`attempts/00N/logs/runner.log`：完整输出；`artifacts/training/code-commit.txt`、`code-diff.patch`：代码版本。

<!-- execution-status -->

执行状态：completed；质量状态：未核实。

<!-- /execution-status -->
