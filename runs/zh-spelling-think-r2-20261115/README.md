# 说字写法 think 第二轮（2026-11-15）

**状态**：2026-11-15 准备完毕，ACP 任务 `pt-4hzqvhue` 排队中。

**要回答的问题**：
- 第一轮 think 开时，无语音、目标缺席、英文和 Common Voice 中文都会坏掉。补上这几类数据的 think 开训练后，think 开能不能在全部守门集上和 think 关一样可用？
- 加入合成说字数据后，think 开时照说明写字的能力能不能比第一轮更强？

第一轮的设计和结果见 `../zh-spelling-think-20261113/README.md`。

## 设计

只训 think 组。第一轮已经比过有无 think 的差别，对照组不再重跑。

| | 第一轮 think 组 | 第二轮 |
|---|---|---|
| 起点 | 11.06 交付版（GRPO v10 插值 50%） | 同左 |
| 训练 | 1 节点 8 卡全参 3000 步，学习率 3.55e-6，预热 300 步，余弦降到 1e-6 | 同左，`configs/train-think.yaml` 与第一轮只差路径和名字 |
| 数据 | 交付配方 v7set34 + 说字清洗后的版本 + 说明记录上采样 | 同左，加合成说字数据（见下） |
| think | 说明记录 0.5；扫描过的来源里的普通中文记录 0.03 | 同左；空答案、扫描没覆盖的记录也以 0.03 写固定句 |
| 代码 | `code-v1`（main `f9f12a3`） | `code-v2`（main `f68e180`，多了 #158、#159） |

### 数据

`configs/train-data-think.yaml` 由 `scripts/make_data_config.py` 从第一轮 think 组的数据配置生成。和第一轮只有三处不同：

- **合成说字数据** `context_correction_synth_zh@train-v1-20261114`：
  - `train` 2,984 条（普通识别），`train_ts` 2,917 条（目标说话人识别，带注册音频）。
  - 每条都有 think，其中 39 条引用太长，只按 think 关训练。
  - 按第一轮上采样的规则，每条在 3000 步里约见 3 次：合计 `weight` 5.27，约占样本 3.1%。
  - 制备和抽检见 `/workspace/workspace/mingdong/data/amphiondata/spelling-synth-zh-20261114/README.md`。
- **think 文件**：在第一轮的 11 个文件之后，加上合成数据的 `think.jsonl.gz`。
- **`empty_prob`、`uncovered_prob` 都取 0.03**，与 `ordinary_prob` 相同：
  - `empty_prob` 管答案为空的样本：目标缺席、纯噪声、无语音数据。
  - `uncovered_prob` 管扫描没覆盖的有声样本：英文、西语等其他语种，以及没有 think 文件的中文来源。
  - 三项取同一个值，不带说明的 think 开样本就是在整个配方上均匀抽出来的，不会偏向某一类。
  - 估算 3000 步约 56 万条样本，其中写固定句的约 1.6 万条。有说明的 think 开样本约 1.2 万条（真实约 3,300、合成约 8,800），两者约 1.3:1。
  - `uncovered_prob` 有使用前提：没有 think 文件的中文来源不能讲字怎么写。本配方满足：说字扫描已覆盖配方里全部真实口语中文来源，没覆盖的是朗读类（AISHELL、MAGICDATA、THCHS-30、Common Voice 等），朗读不讲写法。

### 第一轮续训退化的定位

第一轮两组 think 关的守门指标都比交付版差，最明显的是 GigaSpeech 无语音片段的误触发率（5,446 条，参考为空）。

拿已有的全部评测比较后，结论是：这套 SFT 配方的模型本来就在这个水平，交付版低是因为 GRPO 把它压了下来；在交付版上续训 SFT，就把这部分收益冲掉了。原因不在 think。

| 模型 | 类型 | GigaSpeech 无语音误触发 |
|---|---|---|
| 054 avg3（10.28 版，交付版的 SFT 来源） | SFT | 33.79 |
| ts-robust 059 / 062 / 064 / 065 / 066 / 072 / 073 avg3 | SFT | 31.47 / 32.19 / 35.57 / 40.18 / 36.54 / 39.66 / 40.56 |
| GRPO v5–v18 各轮 avg3 与插值 | GRPO | 20.22–30.96 |
| 交付版 v10 插值 50% | GRPO | 27.82 |
| 第一轮对照组 / think 组 3000 步 | SFT 续训 | 38.21 / 35.88 |

- 第一轮新增的误触发大多是短的中文口语（"来来来""嗯""谢谢""对对对"），也就是 SFT 配方里大量对话数据的风格。
- 数字来自 `/workspace/data/ae-runs/<模型>/runs/*/summary.json` 中 `gigaspeech@icefall-20260908:test/asr/English` 的 `false_alarm_rate`。
- 结论：只做 SFT，GigaSpeech 误触发回不到交付版的水平，这一点第二轮也一样。第二轮的守门对照用第一轮 think 组，不用交付版。能否交付另外判断，见"评测"第 3 项。

## 评测（训练后）

最后一个 checkpoint 测以下三项，配置和打分口径都与第一轮相同（`scripts/make_eval_configs.py`、`make_gate_configs.py`、`run_eval.sh` 从第一轮复制）：

1. **上下文纠偏** `open-audio-llm/context-correction-zh@v1-20261018`，think 关、think 开各测一遍。
2. **守门**：think 关跑第一轮的 8 项。中文 TS-ASR 测试集、asrreg、鼎桥机主识别三项再用 think 开各跑一遍。
3. **能否交付**：和交付版做 50% 权重插值，与交付版自己的做法相同。插值模型同样测 1、2 两项，看 think 能力还剩多少、守门能不能回到交付版水平。模型放进 ae 模型缓存（`ae-models.sh`）。

## 判定

- **think 开可用**：同一 checkpoint 上 think 开减 think 关，下列各项的差都不超过 0.5 个点：
  - 中文 TS-ASR 测试集：CER、误触发率、漏识别率；
  - asrreg：中文 CER、英文 WER、GigaSpeech 无语音误触发率。
  - 第一轮这几项的差分别是 −0.03、+7.85、−1.15、+0.38、+23.76、+43.66。
- **说字更强**：think 开纠正成功率高于第一轮 think 组的 11.98%，保持率和误改率不变差。
- **think 关不劣化**：守门指标不比第一轮 think 组 think 关差。差异用配对 bootstrap 判断。
- **能否交付**：看插值模型。守门回到交付版水平，并且 think 开纠正成功率仍高于交付版 think 关，就进入交付评估；否则把结果记下来，交付版不变。

## 启动

```bash
open-audio-llm experiment run --config runs/zh-spelling-think-r2-20261115/experiment.yaml --dry-run
open-audio-llm experiment run --config runs/zh-spelling-think-r2-20261115/experiment.yaml --task train-think
```

在 ACP 上用 inbox 执行器运行，代码用 `shared/code-v2`。方式同第一轮，执行器目录为 `/workspace/workspace/mingdong/scratch/acp-think-r2`。

## 执行记录（2026-11-15）

- **节点**：用户允许最多占一个 ACP 节点。
  - 提交 1 节点 ACP 任务 `pt-4hzqvhue`（`AmphionASR-spelling-think-r2-synth-empty-uncovered-from-v10interp050-3k-1node`），05:57Z 就绪。
  - 节点 IP 10.119.16.229，8 张 A800 都空闲。
  - 任务里跑 inbox 执行器（`.local-tools/asr-holder/dev_runner.py`，目录 `/workspace/workspace/mingdong/scratch/acp-think-r2`）。
- **W&B 转发**：开发容器的转发（10.119.17.252:17920）按 IP 放行。
  - 重启转发，放行名单加入 10.119.16.229，原有 4 个 IP 不变。
  - 从节点经转发访问 W&B 返回 200。
- **预检**（本机 CPU，代码 `code-v2`）：
  - dry-run 通过。
  - 数据预检通过：117 个来源（第一轮 115 个 + 合成 2 个），配额都不为零。
  - 合成数据每 2,000 条的窗口配额为 `train` 32、`train_ts` 31。按约 280 个窗口算，每条约见 3 次。
- **think 分配核对**：用训练配置的 think 表，在合成数据的真实记录上核对。
  - 记录 ID：think 文件的 5,901 个记录 ID 与 `train`、`train_ts` 逐条对上；语种都是 `zh`，任务分别是 `asr`、`ts_asr`。
  - 合成数据：`explained_prob` 设为 1 时，5,862 条拿到自己的 think，与 think 文件里 think_on 的条数相同；其余 39 条只按 think 关训练。
  - 英文、空答案、没扫描的中文 TS 三类：按配置抽 2 万次，写固定句的比例都是 2.9%，与设定的 0.03 一致。
- **训练**：06:03Z 投递 `10-train-think.sh`，attempt 001 开始训练。
