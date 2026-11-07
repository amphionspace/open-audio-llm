# 06-steering：不训练的激活 steering（决策 token）

## 结论

- **固定幅度的 steering 不划算**：能去掉较多误出字的设置，同时会把大量正样本变成漏识别（例如 L18 α=2 dec fixed，跟错人集去掉 130/148 个误出字，但丢掉 798 个正样本）。固定幅度里最好的一组在跟错人集上只有 +10（去掉 13，丢 4）。
- **自适应 steering（α × (1 − P_present)，P_present 来自第 12 层探针）在两个留出集上都是净正收益**，而且明显优于 05 里的探针门控。跟错人集上，门控 t=0.1 时去掉 33 个误出字、丢 112 个正样本。
  - 两个留出集合计最好的是 **L12 α=4 dec adaptive**：跟错人集去掉 43/148 个误出字、丢 19/4,952 个正样本；场景 TS 去掉 5/18、丢 1/1,823。
  - 幅度再大一些（L16–L18 α=4 adaptive）在场景 TS 和 v12 上去掉的误出字更多（场景 TS 13–16/18），但在跟错人集上丢掉 274–436 个正样本，不能推广。
- 收益幅度不大（跟错人集误出字 −29%，正样本 −0.38%），而且**不同集合上的最优层和最优幅度不一致**。方向是在 v12 上拟合的，换分布后最优 α 会移动；目前只能算"可行的弱信号"，不足以直接上线。

## 输入

- Checkpoint：`/workspace/data/ae-runs/amphionasr2-grpo-v16avg3-ckpt400/checkpoint`
- 方向与探针：**只用** v12 全量状态 `../03-states/states-full.npz` 拟合（`directions.npz`、`directions.json`）。
- 评测（AmphionEval run，vLLM 0.18.0 输出用于对照）：
  - 跟错人集 v3（first-other，留出）：`runs/run-20261107T024531Z-76c3ddb1`，9,030 条，4,026 条 target absent
  - 场景 TS（scene-ts，留出）：`runs/run-20261107T023000Z-3809d6d0`，1,900 条，70 条 target absent
  - v12 均衡子集 2,000 条（`--limit 2000 --seed 0`，**同分布参考**，方向就是在 v12 上拟合的）：`runs/run-20261107T022131Z-c6ec67e6`
- 脚本：`../../scripts/steer_decision.py`（`fit` / `run` / `report`）。输入构造直接复用已验证的 `extract_states.py`（`Encoded`、`select`），没有修改原脚本。

## 方法

- **方向**：v_l = 第 l 层残差（决策位置，= `residual[l]`，即第 l 个 decoder 层的输出）在"target absent 且输出为空"样本上的均值，减去在"target present 且输出非空"样本上的均值。排除 HF 与 vLLM 首 token 不一致的样本。取单位方向，**单位步长 = 0.1 × 该层残差范数中位数**。

  | 层 | |v_l| | 残差范数中位数 | 单位步长 | 误出字到"正确拒识"均值的距离（单位步长） |
  |---:|---:|---:|---:|---:|
  | 12 | 7.75 | 51.3 | 5.13 | 0.8 |
  | 14 | 16.3 | 61.5 | 6.15 | 1.4 |
  | 16 | 46.6 | 88.3 | 8.83 | 2.7 |
  | 18 | 143.6 | 161.2 | 16.1 | 5.8 |
  | 20 | 364.5 | 346.0 | 34.6 | 6.2 |

- **干预**：在第 l 个 decoder 层的输出上加 α × 单位步长 × v̂_l。位置有两种：
  - `dec`：只加在决策位置（`language` token）；
  - `tail`：加在最后一个 audio pad 之后的所有位置（`<|audio_end|>`、`<|im_end|>`、换行、assistant 头和 `language`，共 7 个 token）。

  决策 token 就是 assistant 回合的第一个 token，所以"从第一个 assistant token 开始"在只读首 token 时等同于 `dec`。`tail` 改成覆盖 audio 之后的整段文本，作为更强的变体。不往音频位置上加，因为那会改写所有音频表示，不再是决策层面的干预。
- **自适应**：α_eff = α × (1 − P_present)。P_present 用第 12 层探针计算（拟合方式同 `probe_transfer.py`，v12 上训练准确率 0.994），在未干预的前向里于决策位置读出。
- **网格**：层 {12, 14, 16, 18, 20} × α {0.5, 1, 2, 4, 8} × {dec, tail} × {fixed, adaptive}，共 100 组，另加基线和对照。
- **计算**：输入截断到决策 token（因果 LM，决策 logits 不依赖其后内容）。先做一次完整前向，得到基线决策和 KV cache；把 cache 裁到 tail 之前，然后分批（每批 ≤34 组）只重跑 7 个 tail token。读取最终 logits 的 argmax，记录是 ` None` 还是语言 token，以及 None 的 margin。
- **指标**（相对 HF 基线决策）：`fa_removed` = absent 且基线非 None → None；`fa_new` = absent 且基线 None → 非 None；`hits_lost` = present 且基线非 None → None；`misses_fixed` = present 且基线 None → 非 None；`net` = fa_removed − fa_new − hits_lost + misses_fixed。

## 基线核对

| 集 | HF 完整前向 vs vLLM 首 token | tail 重跑基线 vs 完整前向 | 批内对照（α=0）vs 基线 |
|---|---:|---:|---:|
| 跟错人集 | 0.9994 | 0.9993 | 0.9992 |
| 场景 TS | 0.9989 | 0.9995 | 0.9995 |
| v12-2000 | 0.9985 | 0.9995 | 0.9985 |

三组核对都 ≥ 99%。批内对照与基线不一致的约 0.1%，对应跟错人集上约 7 条近平局样本，场景 TS 上 1 条。小于这个量级的差异（例如 `misses_fixed` 1–2 条）属于数值噪声。

## 结果

基线：跟错人集误出字 148、正样本（非 None）4,952、漏识别 52；场景 TS 误出字 18、正样本 1,823、漏识别 7；v12-2000 误出字 29、正样本 979、漏识别 21。

每格为"去掉的误出字 / 变成漏识别的正样本"（`fa_new` 在下列设置中均为 0）：

| 设置 | 跟错人集（148 / 4,952） | 场景 TS（18 / 1,823） | v12-2000 同分布（29 / 979） |
|---|---:|---:|---:|
| **L12 α=4 dec adaptive**（两个留出集合计最优） | **43 / 19** | 5 / 1 | 15 / 4 |
| L12 α=2 dec adaptive | 21 / 10 | 2 / 0 | 9 / 3 |
| L18 α=1 dec adaptive | 44 / 30 | 5 / 0 | 12 / 2 |
| L20 α=1 dec adaptive | 41 / 25 | 3 / 0 | 12 / 2 |
| L18 α=0.5 tail adaptive | 32 / 17 | 1 / 0 | 9 / 2 |
| L16 α=4 dec adaptive（场景 TS 最优） | 86 / 274 | **13 / 3** | 23 / 6 |
| L18 α=4 dec adaptive（v12 最优） | 92 / 346 | 14 / 6 | **28 / 8** |
| L12 α=0.5 dec fixed（固定幅度中跟错人集最优） | 13 / 4 | 0 / 0 | 3 / 4 |
| L18 α=2 dec fixed | 130 / 798 | 10 / 6 | 20 / 51 |
| 对照：05 探针门控 t=0.1 | 33 / 112 | 11 / 12 | — |

全部 100 组见 `steering.json`。

## 注意事项

- 只看首 token 决策。被 steering 但仍输出语言 token 的样本，后续转写是否改变没有评估；`tail` 变体会改动 assistant 头的 KV，对转写的影响可能更大。
- 跟错人集里"正样本"指 target present 且基线非 None，不区分转写内容是否正确。
- 各集上的最优层和 α 不一致（跟错人集偏好 L12 小幅度，场景 TS 和 v12 偏好 L16–18 大幅度）。L12 α=4 adaptive 是按两个留出集合计挑出来的，因此对这两个集本身有轻微选择偏差。
- 部署需要在 vLLM 里给决策位置的第 l 层残差加向量，还要读第 12 层探针；目前的数字是 HF bf16 eager 的离线结果。
- 运行时间：3 张共享的 A800 并行，单卡约 3 samples/s（每条 1 次完整前向 + 4 次 tail 前向，每次 7 个 token）。跟错人集 3,201 s，场景 TS 566 s，v12-2000 542 s。
- W&B：本机访问 `api.wandb.ai` 返回 403（走代理和不走代理都一样），远端同步未完成。离线 run 在 `../../tracking/wandb/offline-run-20261107_012756-3koa93zt`，网络恢复后用 `wandb sync` 补传。

## 文件

`directions.npz/json`（方向、单位步长、探针），`steer-{first-other,scene-ts,v12-2000}.npz`（每条样本在每组设置下的决策和 None margin）、`.summary.json`（核对结果）、`.log`，`steering.json`（汇总）。
