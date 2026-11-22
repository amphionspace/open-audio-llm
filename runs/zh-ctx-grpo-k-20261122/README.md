# K 上的上下文纠偏 GRPO（2026-11-22）

**状态**：2026-11-22 配置已生成，dry-run 通过，判定已在训练前写定，待提交 ACP。

## 为什么做

上下文纠偏实验第 9 问的 K（`runs/zh-ctx-clue-target-20261117`，分支 `feat/ctx-clue-target-think`，“结果：第 9 问”）：
- 配方：J 的配方换终版文本，加普通识别长对话和西语回放，从 11.17 起训。
- **修好的**：
  - 长对话漏人 52 → 4 条；
  - 西语 WER 不差；
  - 交付评测 ctx-v1 7.87 → 2.14\*；
  - 纠偏 v2 比 J 更好。
- **没过的**：
  - 中文 TS 漏识别变多：v12 2.76 → 3.75\*，多出来的是输出空；
  - 西语 TS 误触发 3.2 → 8.8\*。

这两项都是出字判断的校准。依据是上一轮 GRPO（J → avg3，`runs/zh-ctx-grpo-20261120`）：
- 那一轮用交付版四项奖励（准确率、出字判断、热词、格式）加纠偏奖励，把 J 的同类中文 TS 变差大多拉了回来：
  - v12 CER +0.83\* → +0.10，漏识别率 +0.42\* → +0.10，都不再显著；
  - asr-benchmark 英文无语音误出字 39.79 → 24.57。
- 那一轮修不了的长对话漏人和西语，K 已经修好。

所以在 K 上跑同一套 GRPO。

## 输入

- **起点**：K 的 checkpoint-3000，ae 模型缓存 `checkpoint-3000-f28e379337e1`，带联合 MTP 的 `mtp/`。
- **回放**：K 的 SFT 数据配置 `configs/train-data-template-twins-dialog-final-1117.yaml`（上下文纠偏实验）。
  - 去掉 SFT 专用部分（think、batching、validation）。
  - 已是终版文本，134 个来源。
  - 带 15 个西语来源（含西语 TS 的缺席题），热词设置是 es116 的。
- **目标表**：`context_correction_clean_zh@grpo-targets-v2-20261121`，同 J 那一轮。
  - 14 个来源都在 K 的配置里。
  - 五个说明题子集的目标都落在 K 的 include 列表里。
- **代码**：`shared/code-v3` 是指向 `../../zh-ctx-grpo-20261120/shared/code-v3` 的链接（main 15eacbf），与 J 那一轮同一份，不复制。

## 设计

与 J 那一轮（`runs/zh-ctx-grpo-20261120` 的 attempt 002）相同，只换起点和回放：
- **挖题**：在起点上重挖，08 的五条线（普通、漏字、TS、热词、无语音），加两条纠偏线，16 个采样，think 关。
  - 挖题线的来源是中、英文，没有西语线。
  - 西语只在回放里，靠出字判断奖励作用到西语 TS 的缺席题上。
- **GRPO**：
  - 第十轮参数：lr 1e-6、beta 0.04、400 步、8 × 8 rollout；
  - 奖励：交付版四项，加 `qwen3_asr_context`（权重 1.0）；
  - 联合 MTP：权重 0.1、3 步、衰减 0.8、lr 1e-5，从起点的 `mtp/` 接上。
- **候选**：checkpoint 200/300/400 的平均（avg3），同 J 那一轮。不做插值。
- **评测**（同一节点、同一推理栈：code-v3、vLLM 0.32、MTP 草稿 k=2、贪心）：
  - 上下文纠偏 v2，think 关、开：avg3 和起点 K 用同一份代码各跑一遍，配对比较；
  - 27 套交付评测：GRPO 实验 `runs/zh-ctx-grpo-20261120` 的“交付评估”，`scripts/acp-delivery-one.sh kgrpo`。
    - 与 11.17 已有的 run 配对比较：`python3 scripts/compare_delivery.py r1117 kgrpo`、`scripts/scene_asr_drops.py r1117 k kgrpo`。

生成：`python3 scripts/make_configs.py template-twins-dialog-final-1117 /workspace/data/ae-state/model-cache/entries/checkpoint-3000-f28e379337e1 /workspace/data/datasets/context_correction_clean_zh/versions/grpo-targets-v2-20261121/targets.jsonl`。
- 相对 J 那一轮的 `make_configs.py`，多一条规则：已经是终版的来源，记录列表不重写，不往登记过的目标表目录里写文件。

## 判定（训练前写定）

- 候选是 avg3，都看 think 关，配对 bootstrap 95% CI。
- 除注明外，都是 avg3 对 11.17 的交付评测。

1. **第 9 问的四条都满足**（门槛同第 9 问）：
   - 长对话：
     - scene-asr CER 不显著更差；
     - ≥45 秒覆盖率 ≥ 0.90；
     - 一人基本漏掉的对话 ≤ 60 条。
   - 西语：es-asr、ts-es-v1、ts-es-v2、babble-es、crosslingual-es 的 WER 都不显著更差。
   - 纠偏：
     - ctx-v1 CER 显著好于 11.17，且 ≤ 4.64；
     - 纠偏 v2 上，avg3 对起点 K 的 Emilia 准确率、全部保持率，CI 下限都不低于 −1 个点。
   - TS：场景 TS 的 CER 和误出字率，v12 和跟错人的漏识别率、误触发率，都不显著更差。
2. **西语 TS 误触发修回**：ts-es-v1、ts-es-v2、crosslingual-es 的误触发率都不显著差于 11.17。

**只记录**：
- 其余交付项显著变差的；
- think 开的数字；
- 纠偏 v2 的 TS 题输出空的条数；
- v12 目标在场输出空的条数（11.17 364、K 490）。

**结论**：
- **1、2 都满足**：avg3 作为替换 11.17 的候选，打包并整理完整的交付对比表给用户。替换与否由用户定。
- **只有 v12 漏识别率或西语 TS 误触发不满足**：
  - 记录 GRPO 从 K 拉回了多少；
  - 按来源定位；
  - 不做参数扫描。
- **第 9 问修好的项（长对话、西语 WER、纠偏）有一项被 GRPO 弄坏**：按来源定位并记录，不定配方。

## 执行记录

- **ACP 任务** `pt-cwectlc6`（`AmphionASR-ctx-q10-K-grpo400-v10recipe-ctxplan-mine-avg3-ctxv2-delivery27-1node`，节点 10.119.16.243），执行器目录 `/workspace/workspace/mingdong/scratch/acp-ctx-kgrpo`。13:05Z 提交，13:18Z 开始。
  - 提交时本项目另有 1 个节点在跑，合计 2 个。
- **第 1 次（`10-kgrpo`）**：挖题成功，GRPO 启动失败。
  - 13:23Z 普通识别服务就绪，13:46Z 普通识别阶段完成；13:49Z TS 服务就绪，14:19Z TS 阶段完成。
  - 14:21Z 六份计划写好，同 J 那一轮对照如下：

    | 计划 | K | J 那一轮 |
    |---|---|---|
    | 纠偏 | 4,149 | 5,273 |
    | 热词 | 3,653 | 2,808 |
    | TS 缺席 | 4,102 | 4,544 |
    | TS 在场 | 3,015 | 2,632 |
    | 无语音 | 1,274 | 1,707 |
    | 漏字 | 760 | 851 |

  - GRPO attempt 001 在建采样器时报错退出（14:25Z）：`Increase replay.window_samples to include every source`。
    - 原因：K 的回放多了 15 个西语来源，GRPO 又加了两个只挖不放回的来源，共 136 个。回放窗口沿用 SFT 的 2000，最小的来源 chime6 `train_2spk` 只占权重的 0.023%，按比例应分 0.46 个。采样器按最大余数取整后它是 0，于是拒绝启动。
    - K 的 SFT 只有 134 个来源，这个来源的配额刚好舍入成 1，所以 SFT 时没报错。J 那一轮最小的配额是 0.54。
    - 修法：`make_configs.py` 把 GRPO 数据配置的 `replay.window_samples` 设为能整除 `epoch_samples`、且每个来源按比例算出的配额都不小于 1 的最小值，即 5000。各来源的份额不变，SFT 配置不动。
- **第 2 次（`11-kgrpo`）**：只跑 GRPO 和评测，挖题结果沿用第 1 次的。
  - GRPO attempt 002 14:27Z 开始，生效配置 `window_samples: 5000`。
  - W&B 远端核验：[run](https://wandb.ai/1016097967-amphion/open-audio-llm/runs/AmphionASR-ctx-grpo-v10recipe-ctxplan-from-template-twins-dialog-final-1117-3000-textfinal-mtp-8gpu-400steps-002) running，第 10 步已上传 46 项指标，五项奖励都有数。
  - 成功后由 `stop_when_done.sh 11-kgrpo` 停掉节点。

<!-- execution-status -->

执行状态：partial；质量状态：未核实。

开始：未知；结束：未结束；退出码：未知。

<!-- /execution-status -->
