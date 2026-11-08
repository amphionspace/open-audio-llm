# 13 成对相似嗓音损失的 SFT A/B

## 结论

**和"给对照组加 None 偏置"相比，成对损失在同样误报下少漏 42 条（约 −6%），但对相似嗓音这类难负样本的误报没有专门改善。** 它带来一点真实的区分，不只是挪门槛；不过收益来自注册被污染或太短的正样本，不在原本瞄准的"嗓音像目标的人"上。距离"两轮相加"（v12 12.46 / 453 / 324）仍有明显差距。

| v12（AmphionEval：CER / 误报 / 漏识，条） | 073 avg3（起点） | 对照 v64（attempt 076） | 成对 v66（attempt 078） |
|---|---|---|---|
| v12 | 13.97 / 393 / 460 | 14.12 / 438 / 495 | 14.66 / 215 / 770 |
| TS 英文合集 | 14.54 / 661 / 424 | 14.34 / 788 / 395 | 14.91 / 316 / 558 |
| 跟错人 v3 | 21.84 / 149 / 86 | 21.35 / 171 / 63 | 21.44 / 148 / 79 |
| 客户通用 CER | 27.45 | 20.44 | 17.08 |
| asr-babble-bench CER | 2.33 | 2.36 | 2.38 |
| CHiME-4（英文，仅参考） | 6.33 | 6.63 | 6.60 |

**同等误报下的逐族对比**（`scripts/matched_bias_compare.py`，决策状态口径，与 AmphionEval 的条数略有差异）：
- 给对照组的 ` None` 加 +1.0 偏置，误报 250，与成对组的 252 持平。此时：

| | 误报 | 其中难负样本 | 漏识 |
|---|---:|---:|---:|
| 成对 v66 | 252 | 95 | **698** |
| 对照 v64 + 偏置 1.0 | 250 | 97 | 740 |
| 对照 v64（不加偏置） | 524 | 178 | 449 |

- **少漏的地方**：enroll-contam0-pos −16、enroll-short1-pos −6、dyn-overlap-k3 −5、real-aishell4-3plus-pos −5。
- **多漏的地方**：solo-short-far-pos +26，以及几个短句 solo 集各 +3～4。
- **难负样本误报**：short-hard 33 对 35、abs-hard 9 对 10、abs-hard-k1 4 对 6、enroll-short1-neg 3 对 8。但 realreg3-hard 19 对 15、enroll-contam0-neg 11 对 8，合计 95 对 97，基本持平。

**为什么没打中难负样本（推测，未单独验证）：**
- 孪生题来自 aishellmix / librimix 的干净朗读注册，而 v12 的难负样本多是短混合段、真实录音注册（realreg3）或被污染的注册，分布不同。
- 训练时孪生题几十步内就被学会，`pair_twin_none` 达到 0.9 以上，说明这个任务在训练分布上太容易。

## 方法

- **两组设置**：都从 SFT 073 avg3 续训 3000 步（峰值 lr 2e-6，余弦降到 1e-6，warmup 100），MTP 用 073 avg3 的 `mtp/`，数据为 v7set34 加同语料相似嗓音表。唯一差别是 `objective.paired_lookalike`：weight 4，prob 0.5，top 4，margin 2，gap 20（PR #131、#132、#133）。
- **模型与评测**：两组都取第 2000/2500/3000 步做 avg3；core 评测走部署推理栈（vLLM 0.18.0）。
- **逐族对比**：用两组 avg3 在 v12 全量上的决策状态（`scripts/extract_states.py`）。给对照组加偏置，调到总误报与成对组相同，再逐族比较误报和漏识。
- **中途停掉的两次**：成对损失的前两版在训练中暴露问题，按已确认的失败停掉：
  - 075：真注册一侧的项没有下界，margin 被压到 −160，真注册出字比例降到 0.9。
  - 077：只靠"拉开差距"一项，翻成 None 的真注册拉不回来，8–12% 的真注册输出 None。

  两次的产物都保留。

## 记录

| attempt | 配置 | W&B |
|---|---|---|
| 076 | `train-1node-v64-control.yaml` | `AmphionASR-ts-robust-control-nopair-from073avg3-v7set34-3k-codev24-076` |
| 078 | `train-1node-v66-paired-anchor.yaml` | `AmphionASR-ts-robust-paired-lookalike-gap20-anchor-w4-p0.5-from073avg3-v7set34-3k-codev26-078` |
| 075 / 077 | v63 / v65（已停止） | 同名前缀 `-075` / `-077` |

- 配置位于 `open-audio-llm-ts-robust/runs/ts-robust-acp-20261015/configs/`。
- 评测原始输出：`/workspace/data/ae-runs/ts-robust-07{6,8}-avg3/runs/`。
- 决策状态与偏置曲线：本目录下的 `curve-07{6,8}.json`、`matched-078-vs-076.json`。状态文件 `states-*.npz` 每个约 7.5 GB，只放在 `whai:open-audio-llm/runs/whitebox-20261107/tasks/13-paired-ab/`。
