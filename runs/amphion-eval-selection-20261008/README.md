# 独立选模面板与标准评测链路（2026-10-08）

## 结论

- 下一轮选模与结束评测改用独立数据：AliMeeting 官方 Eval（中文）+ NOTSOFAR 官方 dev（英文 3–7 人）组成选模面板，CHiME-6 dev 只作报告；全部登记为 audio-data-contract `meeting_selection_panel@v1-20261008`，选模面板登记为 AmphionEval 评测集 `open-audio-llm/meeting-selection-panel@20261008`（均为分支，待合并）。
- 评测改为标准链路：`open-audio-llm serve`（vLLM 0.18.0 HTTP）+ `ae open-audio-llm meeting`（AmphionEval 分支新增）。MOSS 用官方 vLLM 构建的 `/v1/audio/transcriptions`，由同一 `ae` 打分。
- 一致性：新打分器对旧输出逐位复现 180 秒集的旧分数（差 0.00）。245 条中文集的旧分数口径不同（meeteval，且在 Qwen3-ASR 折叠复读后的文本上打分），新口径高 1.83 个点，几乎全部来自 1 条复读样本；推理路径差 −0.20 个点，与新链路自身重复运行的差（+0.14）同量级。
- 独立面板基线（cpER，越低越好）：起点 8000 步中文 22.12%、英文多人 125.77%（16 条复读）；MOSS 15.64% / 37.73%。

## 面板构成

| 划分 | 来源 | 条数 | 时长 | 说话人数 | 用途 |
| --- | --- | ---: | ---: | --- | --- |
| `alimeeting_dev` | AliMeeting 官方 Eval 8 场，全部 120 秒窗口，8 通道均值 | 138 | 4.21 h | 1–4 | 选模（中文保持） |
| `notsofar_dev` | NOTSOFAR 官方 dev 36 场，每场轮换取 1 台设备的全部 120 秒窗口 | 146 | 3.73 h | 0–7（5–7 人 129 条） | 选模（英文多人） |
| `chime6_dev` | CHiME-6 官方 dev 2 场，全部 120 秒窗口，U01 4 通道均值 | 150 | 4.46 h | 1–4 | 只报告 |

不按难度或长度筛选；窗口、目标和 turns 沿用 `meeting-long-v1-20260920`（AliMeeting）与 `real-meetings-v1-20261008`（NOTSOFAR、CHiME-6）的 120 秒视图。本轮 6–12 人合成选模集（70 条，AISHELL dev 说话人）继续作为多人目标指标。

推理耗时（单卡 A800，gpu_memory_utilization 0.3，eager，并发 16）：起服务约 3.5 分钟；选模面板 + 多人 70 条并行评测 17 分钟，每个 checkpoint 约 21 分钟。训练每 500 步约 20.5 分钟，模板默认两张卡并行评测相邻 checkpoint。

盲点：

- 窗口约 120 秒，短于部署的 180 秒；180 秒能力由结束评测的 `meeting-180s` 覆盖。
- NOTSOFAR dev 12 位说话人中 10 位也在 NOTSOFAR train（test 与 train 无重叠）；下一轮若用 NOTSOFAR train 训练，英文面板不再说话人独立。
- 没有 8 人以上真实会议；CHiME-6 dev 只有 2 场会议，统计意义有限，因此不进选模。
- AliMeeting dev 此前用于 MOSS 对比（statistical-moss-comparison-500），不是从未看过的评测集，但未用于训练和选模。

## 一致性核对（起点 8000 步）

`old` 为实验脚本原分数，`old→ae` 为同一批旧输出改用 AmphionEval 打分，`ae` 为新链路输出。

| 数据 | old | old→ae | ae | 打分差 | 推理差 | 复读 old/ae（翻转） |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 245 条中文（宏平均） | 13.66% | 15.50% | 15.30% | +1.83 | −0.20 | 1 / 1（0） |
| 同上，新链路重复运行 | 15.30% | — | 15.44% | — | +0.14 | 1 / 1（0） |
| 180 秒 中文真实 | 32.51% | 32.51% | 31.33% | 0.00 | −1.17 | 8 / 7（3） |
| 180 秒 英文真实（AMI） | 61.62% | 61.62% | 67.97% | 0.00 | +6.35 | 4 / 3（7） |
| 180 秒 6–10 人合成 | 85.88% | 85.88% | 90.92% | 0.00 | +5.04 | 0 / 0 |
| MOSS 180 秒 中文 / 英文 / 合成 | 16.43 / 22.72 / 19.60% | 同左 | 16.39 / 22.62 / 19.32% | 0.00 | −0.05 / −0.11 / −0.29 | 0 / 0 |

差异来源：

- 打分：245 条集旧分数用 meeteval 在 `parse_asr_output` 折叠复读后的文本上计 cpCER；新口径在原始输出上计（复读保留）。AliMeeting 部分 16.25% → 19.76%，去掉那 1 条复读样本为 15.88%。180 秒集旧打分即 AmphionEval diarization 口径，新打分器逐条复现。
- 请求：AmphionEval 对中英混合记录追加训练时同样使用的 “Keep each speaker's original language…” 提示（`open_audio_llm.data.qwen3_asr`），旧脚本未追加；`synthetic_zh-en` 78.16% → 93.13% 来自这一差别，合成组的 +5.04 几乎全部由它造成。
- 推理数值：离线引擎为 `qwen_asr` 自带的 vLLM 模型类与静态批，服务为 vLLM 内置 Qwen3-ASR 与连续批处理；同一 BF16 权重、TRITON_ATTN、整段编码器注意力。输出通常在第一个时间戳的百分位处分叉（245 条中 71 条去掉时间戳后文本完全相同）。AMI 的 +6.35 来自复读翻转：差值最大的 5 条贡献 4601 个错误单元，超过总差 3019；去掉复读后新链路 55.10%、旧输出 57.58%。
- MOSS：同一服务重新请求，差 ≤0.3 个点，无复读。

因此选模基线必须用新链路重新测量，不能沿用旧脚本的 13.66%；中文保持容差 1 个点约为重复运行差（245 条、单数据集 ±0.5 点）的两倍。

## 独立面板基线

| 模型 | AliMeeting dev | NOTSOFAR dev | CHiME-6 dev（报告） | 复读（面板） | DER（面板） | 说话人数正确 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 起点 8000 步 | 22.12% | 125.77%（去复读 86.76%） | 121.56%（16 条复读） | 1 + 16 | 15.84% / 49.01% | 0.906 / 0.096 |
| MOSS-Transcribe-Diarize | 15.64% | 37.73% | 55.57% | 0 | 5.69% / 15.53% | 0.833 / 0.445 |

起点 8000 步在 6–12 人合成选模集上的新链路基线为 112.51%（0 条复读；旧脚本 121.50%、1 条复读）。

## 下一轮使用

- 模板：[configs/select-next-round.yaml](configs/select-next-round.yaml) 与 [scripts/select_checkpoints.py](scripts/select_checkpoints.py)。填入训练执行目录后在开发机运行；每个 checkpoint 独占一张卡起服务、并行跑各面板（每个面板一个 W&B run），评测后关服务。
- 停止条件：中文保持（`retention`，AliMeeting dev cpER）连续 `patience` 次超出基线 + `tolerance`；或复读条数（全部面板合计）连续 `patience` 次超过基线 + `loops.max_increase`。最佳点在两项都满足的 checkpoint 中取多人目标最低。停止请求写入训练执行的 `stop-request.json`，训练需加载读取它的回调。
- 结束评测：选中的 checkpoint 跑 `meeting-180s` 与 CHiME-6 dev，并与本实验新链路基线逐组比较。
- [configs/select-baseline.yaml](configs/select-baseline.yaml) 是同一脚本只算起点基线的执行（任务 06-baseline-select）。

## 任务与证据

| 任务 | 配置 | 结果 |
| --- | --- | --- |
| [01-build-panel](tasks/01-build-panel/README.md) | build-panel.yaml | 434 条写入 `/workspace/data/datasets/meeting_selection_panel` |
| [02-export-zh245](tasks/02-export-zh245/README.md) | export-zh245.yaml | 245 条转为本地 AudioRecord（执行 002） |
| [03-serve / 03-serve-moss](tasks/03-serve/README.md) | serve-step8000.yaml / serve-moss.yaml | 评测期间运行，结束后停止 |
| [04–05 一致性](tasks/04-consistency-zh245/README.md) | eval-step8000-zh245 / -meeting180、eval-moss-meeting180 | 见上表 |
| [06–07 基线](tasks/06-baseline-select/README.md) | select-baseline、eval-moss-panel、eval-*-chime6 | 见上表 |
| [08-compare-consistency](tasks/08-compare-consistency/README.md) | compare-consistency.yaml | `consistency.json`、逐条 `*-samples.jsonl`（执行 002） |

AmphionEval 从分支 `feat/meeting-selection-scoring`（`amphion-eval-meeting` worktree）源码运行，合并发布后改为固定版本。W&B：entity `1016097967-amphion`，project `open-audio-llm`，run 名前缀 `amphion-eval-selection-20261008-`，每个执行的 `wandb-verification.json` 记录远端状态与实际上传指标。
