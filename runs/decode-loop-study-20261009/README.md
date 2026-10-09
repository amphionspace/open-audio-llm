# 解码端复读抑制与重跑波动（只推理评测）

## 结论

- **A：推理端不能消除复读，不建议改部署解码。** `repetition_penalty` 1.1 把核心 928 条上的复读从 35–49 条降到 18–30 条（5 个模型，降 37–57%），1.05 降到 16–38 条，但都不为 0（MOSS 为 0），而且会在默认解码三次都不复读的样本上新出现复读（`start` 14 条、`realmix1500` 7 条）。代价是中文准确率稳定变差：1.1 下中文真实会议共同子集 cpER 15 个（模型 × 划分）对比中 14 个变差，+0.3～+1.9 点；固定 338 条集 10 个对比全部变差，+0.1～+2.1 点。英文远场（CHiME-6）共同子集 cpER 5 个模型都下降 4.7～12.1 点，但该集重跑极差就有 5–6 点。
- **B：复读条数和排除复读 cpER 的重跑波动都不小。** 默认解码各重跑 3 次：`start` 复读 48/46/52（标准差 3.06），`realmix1500` 40/35/38（标准差 2.52）；复读过的样本中三次都复读的只有 8/106、16/70。排除复读 cpER 的重跑极差：AliMeeting dev 0.06 / 1.07 点，AISHELL-4 test 0.31 / 0.98，AliMeeting test 1.63 / 0.42，NOTSOFAR dev 3.81 / 1.89，AMI 1.41 / 0.92，CHiME-6 2.18 / 2.86（前为 `start`，后为 `realmix1500`）。训练后 checkpoint 的中文波动约 1 点，不是此前估计的 <0.5 点。
- **建议**：部署保持默认解码（不设 `repetition_penalty`）。选模时把“复读条数差 ≤ 6”和“中文排除复读 cpER 差 ≤ 1 点”视为重跑噪声；复读要从模型/数据侧解决。

详细表格见 [03-analyze 执行 001 的 analysis.md](tasks/03-analyze/attempts/001/artifacts/analysis.md)。

## 为什么做

多人会议 SOT 模型（Qwen3-ASR 1.7B 全参微调）评测时有个别样本复读到 4096 token 上限，一条复读能让 cpER 升高 3–5 点，且同一 checkpoint 重跑时复读与否会随机翻转（vLLM 连续批处理）。数据清洗阶段的 Qwen 推理固定用了 `repetition_penalty=1.1`，部署评测没有设置。本实验回答：

- A. 能否在推理端消除复读，同时不损害排除复读后的准确率？
- B. 默认解码下，复读条数和排除复读 cpER 的重跑波动有多大？

## 方法

- 推理栈与选模一致：`open-audio-llm serve`（vLLM 0.18.0 HTTP，`vllm-serving` 环境，BF16、TRITON_ATTN、eager、seed 42、每批 16 条）+ AmphionEval 0.7.0（`amphion-eval` 环境安装的发布版，`ae open-audio-llm meeting`）。不训练。
- 解码设置只改服务端默认参数（`--override-generation-config`）；ae 请求只带 `temperature=0`、`max_tokens=4096`。服务带 `--enable-log-requests`，每个作业结束时解析全部请求的实际 `SamplingParams`，核验 `repetition_penalty` 唯一且等于配置值、请求数覆盖全部样本（[01-decode-runs](tasks/01-decode-runs/README.md)）。19+5 个作业全部通过。
- `frequency_penalty` 与提前终止没有加组：vLLM 0.18 的服务端默认参数只接受 `repetition_penalty`、`temperature`、`top_k`、`top_p`、`min_p`、`max_new_tokens`，ae 0.7.0 请求端也不能传，要加只能改 ae。
- 全部在完整评测集上跑：选模面板 284 条（AliMeeting dev 138 + NOTSOFAR dev 146）、6–12 人合成 70 条、meeting-180s 424 条、CHiME-6 dev 150 条（合称“核心 928 条”）；固定 338 条集跑默认与 `rp1.1`。
- 除 AmphionEval 自带的“排除复读 cpER”（每次执行只排除自己的复读样本）外，另算“共同子集 cpER”：同一模型所有执行都没复读的样本，设置之间、重跑之间在同一批样本上比较。
- 复读判定沿用 AmphionEval：去掉时间戳后同一字符连续 ≥30 次，或 2–10 字符片段连续 ≥20 次。

## 主要结果

核心 928 条复读条数（default 三次重跑只有 `start`、`realmix1500` 有）：

| 模型 | default | rp1.05 | rp1.1 | rp1.1 相对 default/r1 修复/新增 |
|---|---|---|---|---|
| start（起点 8000 合并） | 48 / 46 / 52 | 36 | 30 | 40 / 22 |
| realmix500 | 49 | 38 | 21 | 40 / 12 |
| realmix1500 | 40 / 35 / 38 | 24 | 18 | 32 / 10 |
| lrhalf500 | 38 | 16 | 23 | 32 / 17 |
| lrhalf2000 | 35 | 23 | 18 | 26 / 9 |

对照：默认解码重跑本身相对 r1 也会“修复/新增” 15–33 / 13–33 条（`start` r2 33/31、r3 29/33；`realmix1500` r2 22/17、r3 15/13）。默认三次都复读的稳定样本里，rp1.1 仍复读 1/8（`start`）和 5/16（`realmix1500`）。

rp1.1 相对 default/r1 的共同子集 cpER 变化（点，正值为变差）：

| 模型 | AliMeeting dev | AISHELL-4 test | AliMeeting test | 固定集 AISHELL-4 | 固定集 AliMeeting | NOTSOFAR dev | AMI test | CHiME-6 dev |
|---|---|---|---|---|---|---|---|---|
| start | +0.60 | +1.48 | +1.25 | +1.03 | +1.13 | −7.69 | +1.70 | −5.77 |
| realmix500 | +1.18 | +1.00 | +0.99 | +1.37 | +2.05 | −2.30 | −3.14 | −4.66 |
| realmix1500 | +1.92 | −0.17 | +1.47 | +0.50 | +1.91 | −0.70 | +1.93 | −4.87 |
| lrhalf500 | +1.18 | +0.64 | +0.75 | +1.17 | +0.98 | −1.27 | −2.00 | −7.91 |
| lrhalf2000 | +0.56 | +0.30 | +0.98 | +0.12 | +0.62 | −0.07 | +2.31 | −12.07 |

rp1.1 下 6–12 人合成与 meeting-180s 合成集（10 个划分）共同子集 cpER 平均变化 −0.3～+0.8 点。rp1.05 对中文的影响方向不一（−0.9～+1.6 点），复读减少也不如 1.1 稳定。

rp1.1 还会改变输出形态：执行 002 中 `start` 的一条 NOTSOFAR 输出出现 U+2028 行分隔符、中英混写（如 “In all围围 yes”“answered the龙”）和时间戳后缺空格，是惩罚高频的换行、说话人标签等 token 后的替代输出。

## 任务

| 任务 | 内容 | 配置 |
|---|---|---|
| [01-decode-runs](tasks/01-decode-runs/README.md) | 5 个模型 × 3 种解码设置；`start`、`realmix1500` 默认解码再重跑 2 次 | `configs/study.yaml` |
| [02-fixed-best](tasks/02-fixed-best/README.md) | 固定 338 条集用 rp1.1 各跑一次 | `configs/fixed-best.yaml` |
| [03-analyze](tasks/03-analyze/README.md) | 汇总、共同子集、复读配对与重跑波动 | `configs/analyze.yaml` |

## W&B

entity `1016097967-amphion`，project `open-audio-llm`；每个评测执行一个 run（`decode-loop-study-20261009-<模型>-<设置>-r<次>-<评测集>-001`，共 86 个，远端状态均为 finished），另有汇总 run：

- 作业汇总：https://wandb.ai/1016097967-amphion/open-audio-llm/runs/decode-loop-study-20261009-runs-003
- 固定集 rp1.1：https://wandb.ai/1016097967-amphion/open-audio-llm/runs/decode-loop-study-20261009-fixed-best-001
- 分析：https://wandb.ai/1016097967-amphion/open-audio-llm/runs/decode-loop-study-20261009-analysis-001

## 资源

GPU 1、2（原属已完成的 real-mix 选模）先用；GPU 0、3 在 lr-half 选模执行完成后才用；GPU 4–7 未用。端口 8811–8814。每个服务固定 `--gpu-memory-utilization 0.3`，启动前要求该卡空闲显存 ≥ 30000 MiB。代码快照在 `shared/code/src`（基于提交 5266962）。

## 遗留

- AmphionEval 0.7.0 用 `str.splitlines()` 读 `worker-predictions.jsonl`，模型输出含 U+2028 时该条之后的样本全部判失败（见 01 执行 002）。本实验在驱动中用 `scripts/rescore.py`（调用 ae 自身打分函数）兜底，最终采用的作业没有触发；应在 AmphionEval 修复读取。
- 默认 r1 同时跑了固定集，r2/r3 没有，并发负载不同；这本身就是部署中会遇到的批处理差异。
