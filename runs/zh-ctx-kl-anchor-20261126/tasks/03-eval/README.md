# 03 评测与判定

- **输入**：两臂最后的 checkpoint，模型名 `klreal`、`klsynth`；
  - 11.17 已有的同口径 run：交付 27 组 `r1117`，纠偏 v2 think 关。
- **做法**：`scripts/acp-node.sh eval`。
  - 纠偏 v2（think 关）：`runs/zh-ctx-grpo-k-20261122` 的 `make_eval_configs.py` 和 `run_eval.sh`，输出在 `/workspace/data/ae-runs/ctx-grpo-k-20261122-<名>`；
  - 交付 27 组：`runs/zh-ctx-grpo-20261120/scripts/acp-delivery-one.sh`，输出在 `/workspace/data/ae-runs/ctx-grpo-20261120-delivery-<名>`；
  - 比较：`compare_delivery.py r1117 <名>`，以及纠偏 v2 对 11.17 的配对比较和显式 / 隐式分组。
- **完成条件**：
  - 两臂的纠偏 v2 和交付 27 组全部退出码 0，W&B 远端都是 finished、有指标；
  - 按 README“判定”逐项给出结论。
- **进展**：
  - 19:01Z 第一次评测给臂 A 用了一半处的 checkpoint，1 分钟内停掉，没有产物（见实验 README“执行”）。
  - 19:03Z 起重跑评测：klreal = `real/checkpoint-1557`，klsynth = `real-synth/checkpoint-3795`（两次 run 的快照里都只出现这个路径）。
  - 19:16Z 两臂纠偏 v2 完成；21:54Z 两臂交付 27 组完成，全部退出码 0，节点自动停掉。
- **结果**：
  - **纠偏 v2（think 关），对 11.17 配对比较**（`cmp-ctxv2-r1117-{klreal,klsynth}.json`，`*` = 95% CI 不含 0）：

    | 指标 | 11.17 | klreal | klsynth |
    |---|---|---|---|
    | 目标词没出现过的真实题（1,680 条，判定 2） | 67.20 | 70.60\* [+2.08, +4.77] | 72.92\* [+4.29, +7.26] |
    | 目标词出现过（223 条） | 71.30 | 76.68\* | 78.48\* |
    | Emilia 准确率（1,903 条） | 67.68 | 71.31\* | 73.57\* |
    | 全部挖掘题准确率 | 67.88 | 71.87\* | 74.09\* |
    | 保持率 | 72.86 | 74.80\* | 85.53\* |
    | 过度纠正率 | 3.19 | 3.40 | 2.41 |
    | CER | 7.42 | 6.71\* | 5.42\* |

    - 对照：K（从 11.17 直接 SFT，回放算交叉熵，含全部合成数据）在没出现过的题上是 70.36\*。
  - **逐条翻转**（`flips-r1117-{klreal,klsynth}.json`，`ctx_flip_breakdown.py`，Emilia 1,903 条）：

    | | 新对 | 新错 | 净 | 显式 | 隐式 | 一致性 |
    |---|---|---|---|---|---|---|
    | klreal | 110 | 41 | +69 | +44 | +21 | +4 |
    | klsynth | 162 | 50 | +112 | +102 | +9 | +1 |

    - 合成说字数据主要让显式说明题多改对（+102 对 +44），隐式题反而少（+9 对 +21），和 J 的规律一致（显式 +44 / 隐式 +2）。
  - W&B：klreal <https://wandb.ai/1016097967-amphion/open-audio-llm/runs/run-20261126T191009Z-1a920fb3>，klsynth <https://wandb.ai/1016097967-amphion/open-audio-llm/runs/run-20261126T191009Z-73f145c5>，远端都是 finished、有指标。
  - **交付 27 组**（`runs/zh-ctx-grpo-20261120/tasks/04-delivery/delivery-r1117-vs-{klreal,klsynth}.md`，两臂各 27 个 W&B run 远端 finished、有指标；快照里只出现各自最后的 checkpoint）。
    - 判定 1 的 TS 项（误触发 FA / 漏识别 miss，`*` 显著，**粗体**为显著变差）：

      | 项目 | 11.17 | klreal | klsynth |
      |---|---|---|---|
      | v12 FA | 2.46 | **2.72\*** | **3.65\*** |
      | v12 miss | 2.76 | 2.79 | 2.05\* |
      | 跟错人 FA | 2.58 | **2.96\*** | **4.07\*** |
      | 跟错人 miss | 1.66 | 1.40\* | 1.02\* |
      | 英文 TS FA | 6.20 | 5.85\* | **8.41\*** |
      | 英文 TS miss | 2.94 | **3.28\*** | 1.97\* |
      | 西语 TS v1 FA | 3.21 | 2.56\* | **4.69\*** |
      | 西语 TS v2 FA | 3.03 | 2.44\* | **4.68\*** |
      | 跨语种（西语）FA | 1.04 | 1.04 | **2.29\*** |

    - 显著变差的全部项目：
      - klreal（12 组，18 项指标）：v12 CER、v12 FA；西语识别 WER；西语 TS v1、v2 的 WER 和 miss；babble3、babble4 的 CER 和 WER；英文 TS 的 WER 和 miss；跟错人 FA；asr-benchmark CER；police CER；跨语种 FA；跨语种（英文）miss。
      - klsynth（14 组，20 项指标）：v12 FA；西语识别 WER；西语 TS v1、v2 FA；babble3、babble4 的 CER 和 WER；英文 TS FA；跟错人 CER 和 FA；asr-benchmark CER、热词命中率、英文 FA；police CER 和热词命中率；跨语种 FA；长对话 scene-asr CER（4.72 → 5.88\*）；跨语种（英文）FA；跨语种（西语）FA。
    - 纠偏相关的交付项都变好：ctx-v1 CER 7.87 → 6.65\*（klreal）/ 5.76\*（klsynth），chime4 也显著变好。
  - **TS 回退的诊断**（判定 1 不成立的分支）：逐条比较“出字 / 不出字”的翻转（同一批 predictions，11.17 对候选）。

    | 目标不在场：新出字 / 新不出字 | klreal | klsynth |
    |---|---|---|
    | v12 | 84 / 49 | 187 / 23 |
    | 跟错人 | 24 / 9 | 63 / 3 |
    | 英文 TS | 58 / 95 | 256 / 22 |
    | 西语 TS v2 | 27 / 78 | 166 / 24 |

    - 两臂的 TS 回退都是“在场判断”在动：`language` 后面那个 token 选 `None` 还是语种。输出的语种标记都对，新出字的内容是旁人的整段话，集中在 1 秒注册音、注册音被污染、短难例这类临界样本上。
    - klsynth 所有语种都往出字偏；klreal 中文往出字偏，英文、西语往不出字偏（西语 v2 目标在场：新不出字 88、新出字 10）。
    - 回放 KL 最后 10% 各任务：TS 正样本最高（0.018 / 0.019），TS 负样本只有 0.006 / 0.004。KL 是对全部监督 token 取平均，负样本的判断 token 在一个回放 batch 里只占很小的份量。
    - 根因：CE 流的每条样本都是 `language <语种><asr_text>正文`，没有一条是 `None`，CE 一直在直接训练这个判断 token；稀释后的 KL 拉不住临界样本。合成数据又多了 2,984 条“总是出字”的样本，所以 klsynth 偏得更多、方向也一致。
- **判定**（README“判定”）：
  1. TS 保住：**两臂都不成立**（见上表）。
  2. 真实录音纠偏收益：**两臂都成立**。没见过的真实题 klreal +3.4\*、klsynth +5.7\*，都超过 K 的 +3.2\*。
  3. 交付评测：两臂都有多项显著变差，都不是替换候选。
- **结论**：回放 KL 保住了大部分能力，回退比之前的续训小，但 CE 流直接训练在场判断 token，TS 校准还是会偏。纠偏收益在正文 token 上，在场判断不需要由 CE 流来训。
