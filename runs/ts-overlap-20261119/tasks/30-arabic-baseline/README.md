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
- 打分：共享的 amphion-eval 环境是 0.13.3，没有这些评测集，也没有阿语打分规则（78c1583）。所以评测程序用固定快照 amphion-eval ca7d87f（原生打分 v6），装在 `/workspace/data/ts-overlap-20261119/ar-baseline/amphion-eval-0.15.0-ca7d87f`，通过 `AE_PATH` 放到 PYTHONPATH 最前面。共享环境不动。
- 作业：[job_ar_baseline.sh](../../scripts/job_ar_baseline.sh)，1 节点，4 路并行。输出 `/workspace/data/ts-overlap-20261119/task30-ar-r1117/`，评测记录在 `/workspace/data/ae-runs/whitebox-merge-v18a-es116inc/`。

## 完成条件

7 个评测都 rc=0、覆盖率 100%，W&B 远端 finished 且指标已上传。这是基线测量，不设达标门槛。

## 进展

- 2026-11-28：准备配置和固定快照，提交前通知其他会话。
- 2026-11-28 09:40–09:54 UTC：ACP `pt-ozm00eyi`（节点 10.119.16.253）跑完，7 个评测都 rc=0，覆盖率都是 100%（失败 0 条），W&B 远端 7 个 run 都是 finished、指标已上传。**执行完成。**

## 结果

**结论：11.17 不会阿拉伯语。** 词错率普遍超过 100%。它多半把阿语判成西班牙语或英语：普通识别 23,109 条里，判成西班牙语的 11,934 条、英语 9,495 条、中文 1,445 条，判成阿语的只有 133 条。所以识别类数字只说明"没有能力"，不能用来比较细节。可以直接用作守门基线的是负样本的误报和热词召回。

| 评测（run） | 主要数字 |
|---|---|
| ar-asr-test（`run-20261128T094343Z-853958e6`，23,109 条） | 词错率：CV 128.5%，FLEURS 109.4%，MGB-2 75.0%，Casablanca 8 种方言 83.3–109.2% |
| asr-babble-bench-ar（`095129Z-4392e272`，1,400 条） | 干净 166.3%，babble 20/10/5 dB 123.2/124.6/129.2%；纯 babble 误报 远 1.0%、中 1.5% |
| ts-asr-benchmark-ar v1（`094413Z-66fc709b`，13,440 条） | 目标在场 123.9–133.8%；目标不在场误报 0–8.7%（注册音被污染时最高：contam0 8.7%、contam10 6.0%） |
| ts-asr-benchmark-ar v2（`094458Z-438866d9`，17,160 条） | v1 各列与上面相同（差 ≤0.02）；补充列：dyn 127.5–222.3%，abs-k1/k2/k4/hard 误报 0.8/4.2/6.7/8.3%，非语音 6 类误报都是 0 |
| ts-asr-crosslingual-ar（`095017Z-ce78b98a`，1,200 条） | 在场 112.6–145.3%；跨语注册时漏识 23–30%（同语 3–7%）；不在场误报 0–1.7% |
| hotword-bench-ar（`094613Z-5607e18a`，11,192 条） | 不给热词召回：CV 7.1%、FLEURS 19.9%、MGB-2 28.6%；给标准热词召回：CV 14.0%、FLEURS 34.5% |
| ts-asr-hotwords-ar（`095047Z-fa2f8239`，7,558 条） | 合成集召回：不给 3.5%，随机 9.4%，干扰人 9.0%；误报 ≤0.13% |

评测记录在 `/workspace/data/ae-runs/whitebox-merge-v18a-es116inc/runs/`，日志在 [configs/eval-r1117/logs/](../../configs/eval-r1117/logs/)。

## 对后续阿语训练的意义

- 阿语训练是从零学一门语言，不是在已有能力上修补。所以第一轮的判断标准是：阿语词错率降到可用水平，同时中英文和任务 29 的交付守门不回退。
- 目标不在场的误报（TS v2 的 abs-hard 8.3%，注册音被污染时 8.7%）现在偏低，是因为模型本来就很少输出阿语。训练后这两项反而可能上升，要单独盯。
- 跨语注册的漏识（23–30%）是说话人选择的问题，不依赖阿语识别能力。可以作为训练后说话人选择是否保住的对照。
