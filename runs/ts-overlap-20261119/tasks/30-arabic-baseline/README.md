# 30 阿拉伯语：11.17 基线

## 要回答的问题

交付线按"重叠场景 → 阿拉伯语 → 葡萄牙语"推进（用户 11-28 定的节奏）。加阿拉伯语之前，先量出 11.17 在阿语评测上的水平，作为后续阿语训练的对照和守门基线。

## 输入

- 模型：11.17 交付包，缓存名 `whitebox-merge-v18a-es116inc`（条目 `model-afea989da22b77adc0ce9365`）。服务栈和交付守门相同：[eval_lanes.sh](../../scripts/eval_lanes.sh)，pack-mtp `serve.sh`，不开 MTP。
- 评测集（AmphionEval MR !174，均为 `@v1-20261127`，由 audio-data-contract-d9 建）：
  - `asr/ar-asr-test`：CV 24.0 test、FLEURS ar_eg、MGB-2 test、Casablanca 8 种方言；
  - `asr/ts-asr-benchmark-ar` v1 / v2：v2 在 v1 基础上加了补充集各列；
  - `asr/ts-asr-crosslingual-ar`；
  - `asr/hotword-bench-ar`；
  - `asr/ts-asr-hotwords-ar`；
  - `asr/asr-babble-bench-ar`。
- 配置：[configs/ar-templates/ar/](../../configs/ar-templates/ar/)，取自 d9 的冒烟配置（`/workspace/workspace/mingdong/data/amphiondata/ar-benchmarks-20261127/smoke/`），只把 `samples_per_source` 改成 0（全量）。不发布。
- 打分：共享的 amphion-eval 环境是 0.13.3，没有这些评测集，也没有阿语打分规则（78c1583），共享环境不动。所以评测程序用固定快照 amphion-eval ca7d87f（原生打分 v6），装在单独的 venv `/workspace/data/ts-overlap-20261119/ar-baseline/ae-ca7d87f-venv`：用 `--system-site-packages` 继承共享环境的包，再装 ca7d87f 的 wheel。[eval_lanes.sh](../../scripts/eval_lanes.sh) 用 `AE_ENV` 指向它。
  - 不能用 PYTHONPATH 叠加：after-training 启动评测子进程前会删掉 PYTHONPATH（`amphion_eval/integrations/open_audio_llm/after_training.py:473`），推理和打分就退回共享环境的 0.13.3。第一轮就是这样，见"进展"。
- 作业：[job_ar_baseline.sh](../../scripts/job_ar_baseline.sh)，1 节点，4 路并行。评测记录在 `/workspace/data/ae-runs/whitebox-merge-v18a-es116inc/`。

## 完成条件

7 个评测都 rc=0、覆盖率 100%，按 v6 打分，W&B 远端 finished 且指标已上传。这是基线测量，不设达标门槛。

## 进展

- 2026-11-28：准备配置和固定快照，提交前通知其他会话。
- 2026-11-28 09:40–09:54 UTC，第一轮：ACP `pt-ozm00eyi`（节点 10.119.16.253），7 个评测都 rc=0、覆盖率 100%、W&B finished。**但打分不对**：run.json 记的口径是 "v3"，没有阿语规则。原因就是上面的 PYTHONPATH 被删。这一轮的 7 个 run（`run-20261128T094343Z-853958e6`、`094413Z-66fc709b`、`094458Z-438866d9`、`095017Z-ce78b98a`、`095129Z-4392e272`、`094613Z-5607e18a`、`095047Z-fa2f8239`）作废，只作为第二轮复用的推理结果。
- 2026-11-28 10:46–10:56 UTC，第二轮：改用 venv 后，在开发机 GPU 4–7 上重跑（输出 `/workspace/data/ts-overlap-20261119/task30-ar-r1117v6/`，日志在 [configs/eval-r1117v6/logs/](../../configs/eval-r1117v6/logs/)）。
  - 7 个评测都 rc=0，覆盖率 100%，summary 的 protocol version 是 6。W&B 远端 7 个 run 都是 finished、指标已上传。**执行完成。**
  - 全部回答都复用了第一轮的推理。0.15.0 只在推理参数一致、样本行（记录、上下文、参考文本）完全相同时才复用。
  - 两个版本拼请求的代码（`models/openai_audio.py`、`prompts.py`）完全相同。输出解析 `qwen3_output.py` 只新增了思考模式用的 `split_think`，这里没用到。所以第二轮就是在同一批回答上按 v6 重新打分。

## 结果（v6 打分）

**结论：11.17 不能直接用于阿语，主要坏在语种判别。**
- 普通识别 23,109 条里，语种标签判成西班牙语的 11,934 条、英语 9,495 条、中文 1,445 条，判成阿语的只有 133 条。
- 但只要输出的是阿拉伯文，转写就比较准，说明底座 Qwen3-ASR 的阿语识别能力还在：

| 来源 | 输出是阿拉伯文的比例 | 这部分词错率 | 其余部分词错率 |
|---|---|---|---|
| CV | 17% | 38.1% | 148.1% |
| FLEURS | 60% | 22.9% | 246.9% |
| MGB-2 | 64% | 30.0% | 161.1% |
| Casablanca 8 种方言 | 38–73% | 53.4–89.1% | 121.8–167.0% |

各评测的主要数字：

| 评测（run） | 主要数字 |
|---|---|
| ar-asr-test（`run-20261128T105013Z-4e72dc99`，23,109 条） | 词错率：CV 122.8%，FLEURS 106.4%，MGB-2 73.3%，Casablanca 79.4–110.9% |
| asr-babble-bench-ar（`105241Z-b7b521cc`，1,400 条） | 干净 161.7%，babble 20/10/5 dB 117.3/120.4/126.4%；纯 babble 误报：远 1.0%、中 1.5% |
| ts-asr-benchmark-ar v1（`105133Z-4947dd32`，13,440 条，106 列） | 在场合计词错率 139.1%，平均漏识 7.7%；不在场合计误报 1.96%。误报最高的是注册音被污染（contam0 8.7%、contam10 6.0%）和 abs-hard 8.3% |
| ts-asr-benchmark-ar v2（`105043Z-af5a0b0e`，17,160 条，137 列） | 在场合计 139.1%；不在场合计误报 2.26%（8,640 条）。v2 比 v1 多 31 列：events、crowd、rir、mm、abs 等，其中 mmfarfar-neg 误报 11.7% 最高 |
| ts-asr-crosslingual-ar（`105521Z-702efa56`，1,200 条） | 在场 112.2–145.4%；跨语注册时漏识 23.3–30.0%（同语 2.5–6.7%）；不在场误报 0–1.7% |
| hotword-bench-ar（`105151Z-155be038`，11,192 条） | 不给热词时召回：CV 7.1%、FLEURS 19.9%、MGB-2 28.6%；给标准热词：14.0/34.5/44.8% |
| ts-asr-hotwords-ar（`105351Z-fc379ffb`，7,558 条） | 合成集召回：不给热词 3.5%，随机列表 9.4%，含干扰人 9.0%；误报 ≤0.1% |

## 对后续阿语训练的意义

- 训练的首要目标是让模型输出 "language Arabic" 再转写，而不是从零学阿语。
  - 训练代码原来把 `supervision.language` 的 "ar" 原样写成 "language ar"，现已映射到原生标签 "Arabic"（oal-clean-teacher e3dd43b）。
- 第一轮阿语训练的判断标准：阿语词错率降到可用水平，同时中英文和交付守门不回退。
- 跨语注册的漏识（23–30%）、注册音污染和 mmfarfar 的误报属于说话人选择问题。训练后要单独对照这几项。
