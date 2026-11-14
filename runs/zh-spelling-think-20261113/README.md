# 说字写法 think 第一轮（2026-11-13 准备，2026-11-14 启动）

**状态**：2026-11-14 两组都训完 3000 步并评测完毕。按判定条件 think 有用；但两组续训都让几项守门指标比交付版变差，按条件先排查续训本身（见"结果与结论"）。执行细节见文末"执行记录"。

**要回答的问题**：数据和步数相同时，加入 think 开关训练，能不能让模型在 think 开时更会照说话人的说明写字（"zha 这个音，说话人说是「山楂的楂」，写作「楂」"）；think 关时，普通识别不劣于不加 think 的对照组。

## 设计

| | think 组（`tasks/01-train-think`） | 对照组（`tasks/02-train-control`） |
|---|---|---|
| 起点 | 11.06 交付版（GRPO v10 插值 50%）`/workspace/data/ae-runs/amphionasr2-grpo-v10interp050/checkpoint` | 同左 |
| 训练 | 1 节点 8 卡全参 3000 步，学习率 3.55e-6，预热 300 步，余弦降到 1e-6；其余同 SFT 073（v62） | 同左 |
| 数据 | 交付配方 v7set34 + 说字清洗后的版本 + 说明记录上采样（见下） | 同左 |
| think | `think` 一节：有说明的记录以 0.5 开 think，同来源的普通中文记录以 0.03 写固定句 | 无 |

**唯一的变量是数据配置里的 `think` 一节**（`configs/train-data-think.yaml` 与 `train-data-control.yaml` 只差这一节）。

两组都换成清洗后的版本，并且同样上采样说明记录。这样两组的差别只来自 think，不来自数据清洗或采样。

**与 SFT 073 的其他差别**
- **起点**：用交付版，所以结果能直接和线上比（用户 11-13 选定）。
- **步数**：3000 步（073 是 10000 步），属于续训：交付版已做过 GRPO，SFT 越长，离交付版越远。
- **不带联合 MTP**：交付版没有 MTP 模块，MTP 也不是本轮要比的变量。
- **TS 提示**：代码快照用 11-06 起的新 TS 提示（`TS_SYSTEM`），交付版训练时用的是旧提示。两组一样，但和交付版对比 TS 指标时要记住这一点。

**代码**：`shared/code-v1` 是 open-audio-llm main `f9f12a3` 的快照，包含：
- #149：think 训练；
- #150：think 预填的服务模板；
- #151：`include_records`；
- #152、#153：replay KL 的两处修复（别的实验合入，两组同样带上）；
- #154：建单个来源的索引时跳过 think 表（第一次预检因此报错）。

### 数据

- **清洗后的版本**：交付配方里 11 个中文来源的版本，换成 `context_correction_clean_zh@v1-20261110` 判定后的派生版本 `<版本>-ctxclean-20261110`。
  - 交付配方用的是带标点、带 utts 时间线的版本，判定按字对齐迁到这些版本上（audio-data-contract `clean.py transfer`）。迁移过程和结果见运行目录 `/workspace/workspace/mingdong/data/amphiondata/ctx-clean-zh-20261110/README.md`"迁到交付配方的版本"一节。
  - alimeeting_sdm 两个版本没有改动，照旧用原版本。
- **think 文本**：`context_correction_clean_zh@think-v2-20261113`，每个来源一个 `think.jsonl.gz`。
  - 共 2,187 条记录带 think：WenetSpeech 923、Emilia2 mix 642、Emilia2 对话 289、Emilia2 短句 281、会议 33、VITW 19。
  - 另有 32,346 条只按 think 关训练。
- **说明记录上采样**：
  - 每个说明记录较多的 split 再加一个来源，用 `include_records` 只保留其中带 think 的记录，权重按"每条在 3000 步里约见 3 次"算（`scripts/make_data_configs.py`）。
  - 一共 5 个来源，合计 `weight` 1.80，约占样本的 1.1%：
    - wenetspeech_clean `train` 923 条；
    - emilia2_mix utts `train_2spk` 326、`train_3spk` 316；
    - emilia2_dialog `train_2spk` 229；
    - emilia2_short `train_zh` 281。
  - 按 073 实测，每步约 188 条样本（10000 步消耗 0.94 × 2,000,000），3000 步约 56 万条。
  - 少于 100 条的 split（会议各版、VITW）不单独上采样：在 2,000 条的 replay 窗口里配额会是 0。
- **`ordinary_prob` 的取值**：普通 think 开样本大致和说明 think 开样本一样多（约 1:1，同 arXiv 2606.10838 的 50/50）。说明样本只有 2,187 条，取 `explained_prob` 同样的值的话，think 开几乎全是固定句，模型会学成一律写固定句。

### 预检（2026-11-13，CPU，只查数据）

- think 组 `python -m open_audio_llm.data.catalog_cache --data_config configs/train-data-think.yaml --batch-size 12 --world-size 8 --preflight-report …` 通过。
  - 115 个来源（交付配方 110 个 + 上采样 5 个），配额都不为零。
  - think 表的 11 个数据集都有训练来源在读。
  - 第一次预检报错：建单个来源的索引时也会检查 think 表，open-audio-llm #154 已修。
- 上采样来源每 2,000 条窗口的配额：WenetSpeech 10、Emilia2 mix 2spk/3spk 各 3、Emilia2 对话 2、Emilia2 短句 3。按 3000 步约 280 个窗口算，每条说明记录约见 2.4–3.0 次（另外在原来源里还会出现）。
- 真实数据抽查：把 `explained_prob` 设为 1，上采样的 2,075 条记录全部拿到了自己的 think。说明 think 表的记录 ID 与加载器一致，不会悄悄全走 think 关。示例：
  `<think>\njia 这个音，说话人说是「单人旁两个土的佳」，写作「佳」。\nying 这个音，说话人说是「聪颖的颖」，写作「颖」。\n</think>\n\nlanguage Chinese<asr_text>嗯，大家好，我叫李佳颖，单人旁两个土的佳，然后聪颖的颖，然后。`

## 评测（训练后）

- **上下文纠偏**：`open-audio-llm/context-correction-zh@v1-20261018`。
  - 每个 checkpoint 测 think 关和 think 开两遍；think 开要求服务带 think 模板，评测带 `think: true`。
  - 配置用 `scripts/make_eval_configs.py <名字> <checkpoint> <ASR GPU> <TS GPU> <端口>` 生成。交付版的配置已生成在 `configs/eval-release-v10interp050/`，用来确认带 think 模板的服务在 think 关时与线上同分（交付版上下文纠偏 8.88）。
  - 评测环境要用含 AmphionEval !147、!149 的版本（main `3a5a9c1` 之后）；`amphion-eval` 环境里装的 0.13.3 还没有 `--think`。
- **守门**：沿用 SFT 系列的标准评测，与交付版、对照组同口径比较（think 关）。
  - v12 TS-ASR、普通 ASR 回归（asrreg）、热词、警务、CHiME-4、鼎桥客户录音；
  - 生成方式同 `runs/ts-robust-acp-20261015/configs/eval-066-avg3-*`。

## 判定与停止条件

- **think 有用**：think 组 think 开的纠正成功率明显高于对照组，同时保持率（误改率）不变差；另外 think 组 think 关不低于对照组。
- **think 没用**：think 开不高于对照组 → 按 2026-11-12 的判断，不再加更丰富的 think 文本，改为只做清洗数据的 SFT。
- **普通识别变差**：任一组 think 关的守门指标比交付版明显变差 → 先排查续训本身（对照组也会变差），不算 think 的问题。
- 两组都训完 3000 步后，评测最后一个 checkpoint；中间每 500 步存一个，需要时补评。

## 结果与结论（2026-11-14）

**结论**
- **按判定条件，think 有用。**
  - think 组 think 开的纠正成功率 11.98%（968 条中 116 条），对照组 8.16%（79 条）。
  - 保持率 78.41% 对 76.14%，误改率 5.58% 对 5.79%，都没有变差。
  - think 组 think 关不低于对照组：合并 CER 7.25 对 7.59，配对 bootstrap 差 −0.34，95% CI [−0.64, −0.02]。
  - 守门指标（think 关）两组基本持平，见下表。
- **同时触发"普通识别变差"。**
  - 两组 think 关都有几项守门指标明显比交付版差，对照组同样如此，所以是续训本身的问题，不是 think 的问题。
  - 主要几项：GigaSpeech 无语音片段误触发 +10.4 个点；中文 TS-ASR 测试集 CER +0.5、误触发 +0.8；上下文纠偏合成 TS 子集 +3.2；鼎桥警务指令（内置热词）CER 15.40 → 18.35。
  - 按条件，下一步先排查续训，再决定这条线能否交付。
- **两点说明。**
  - 对照组和交付版 think 开都不可用：模型在预填的 `<think>` 之后不闭合，答案为空，CER 100%。所以 think 开只能和对照组 think 关比。
  - think 开的合并 CER 降了 2.77，大部分来自合成 TS 子集漏识别减少（miss 5.36 → 2.40）。照说明改字本身的提升是纠正成功率 +3.8 个点，绝对值仍然低。`ae open-audio-llm compare` 不给纠正成功率的置信区间，这里只报条数。

### 上下文纠偏

测试集 `open-audio-llm/context-correction-zh@v1-20261018`，AmphionEval 0.14.0（新 TS 提示，与两组训练一致）。合并 CER = 三个子集错误数之和 / 参考字数之和。

| | 合并 CER | 合成 | 合成 TS | 真实挖掘 | 纠正成功率 | 保持率 | 误改率 | 真实挖掘准确率 |
|---|---|---|---|---|---|---|---|---|
| 交付版 think 关 | 6.19 | 2.58 | 10.09 | 5.62 | 6.61 | 76.96 | 4.96 | 68.50 |
| 对照组 think 关 | 7.59 | 2.46 | 13.24 | 5.34 | 8.16 | 76.14 | 5.79 | 77.17 |
| think 组 think 关 | 7.25 | 2.46 | 12.52 | 5.30 | 8.88 | 76.65 | 5.79 | 77.95 |
| think 组 think 开 | **4.81** | 2.66 | **7.07** | 5.25 | **11.98** | 78.41 | 5.58 | 80.31 |
| 交付版、对照组 think 开 | 100 | 100 | 100 | 100 | 0 | — | — | — |

- 交付版这一行用新 TS 提示，所以合并 CER 是 6.19，不是线上的 8.88。线上 8.88 是旧提示；旧提示客户端重跑为 8.85，见"带 think 模板的服务核对"。
- 配对 bootstrap（`ae open-audio-llm compare`，差 = 后者 − 前者，* 表示 95% CI 不含 0）：

| 对比 | 合并 CER | 合成 | 合成 TS | 真实挖掘 |
|---|---|---|---|---|
| 对照组 → think 组 think 关 | −0.34 [−0.64, −0.02]* | −0.01 | −0.72 [−1.36, −0.06]* | −0.04 |
| 对照组 → think 组 think 开 | −2.77 [−3.46, −2.10]* | +0.19 [−0.06, +0.51] | −6.16 [−7.60, −4.70]* | −0.08 |
| 交付版 → 对照组 | +1.40 [+0.96, +1.85]* | −0.11* | +3.15 [+2.22, +4.13]* | −0.28* |

- think 开的输出：3249 条中 3171 条正常闭合 think。
  - 1969 条写成说明格式，例如"yu 这个音，说话人说是「就是多余的余」，写作「余」。"
  - 1279 条写固定句。
  - 说明 think 训练确实生效了。

### 守门（think 关）

| 评测 | 指标 | 交付版 | 对照组 | think 组 | 对照组 − 交付版 | think 组 − 对照组 |
|---|---|---|---|---|---|---|
| 中文 TS-ASR 测试集 `ts-asr-benchmark@v12-20261022` | CER | 12.88 | 13.36 | 13.41 | +0.48 [+0.23, +0.70]* | +0.06 [−0.15, +0.25] |
| 〃 | 误触发率 | 2.69 | 3.49 | 3.25 | +0.80 [+0.58, +1.01]* | −0.23 [−0.40, −0.07]* |
| 〃 | 漏识别率 | 2.85 | 3.08 | 2.96 | +0.23 [+0.06, +0.41]* | −0.12 [−0.27, +0.04] |
| 普通 ASR 回归 asrreg（中文、方言） | CER | 3.56 | 3.67 | 3.72 | +0.11 [+0.09, +0.13]* | +0.05 [−0.01, +0.16] |
| 〃 | 术语命中率 | 98.83 | 98.79 | 98.70 | −0.03 [−0.13, +0.06] | −0.09 [−0.16, −0.03]* |
| asrreg（英文） | WER | 7.33 | 7.42 | 7.44 | +0.09 [+0.05, +0.13]* | +0.02 [−0.01, +0.04] |
| 〃 GigaSpeech 无语音片段 | 误触发率 | 27.82 | 38.21 | 35.88 | +10.39 [+9.64, +11.25]* | −2.33 [−2.83, −1.88]* |
| 警务全量 | CER | 0.26 | 0.27 | 0.27 | +0.01 [−0.00, +0.02] | −0.00 [−0.01, +0.00] |
| 鼎桥客户普通识别 `customer-asr@v5` | CER | 18.39 | 17.37 | 18.69 | −1.02 [−3.08, +0.68] | +1.31 [−0.32, +3.25] |
| 鼎桥客户机主识别 `customer-ts@v4` | CER | 6.25 | 3.12 | 4.17 | −3.12 [−5.41, +0.00] | +1.04 [+0.00, +2.94] |
| 热词 `qwen3-asr-hotwords@20261001` aishell 带热词 | CER / 召回 | 0.34 / 99.85 | 0.33 / 99.93 | 0.33 / 100.00 | — | — |
| 〃 commonvoice_en 带热词 | WER / 召回 | 4.91 / 98.23 | 5.26 / 98.30 | 5.23 / 98.30 | — | — |
| 鼎桥客户热词：警务指令，内置热词（137 条） | CER / 召回 | 15.40 / 77.37 | 18.35 / 75.18 | 18.78 / 74.45 | — | — |
| 鼎桥客户热词：全部 | 错误率 / 召回 | 14.68 / 72.22 | 16.13 / 72.22 | 16.81 / 69.79 | — | — |
| CHiME-4 et05 real（英文，仅参考） | WER | 7.70 | 7.56 | 7.03 | −0.14 [−0.87, +0.56] | −0.53 [−1.07, −0.07]* |

- **口径**
  - 交付版的数字来自 11-05/11-06 的 run，当时用旧 TS 提示客户端。两组用 0.14.0，即新 TS 提示，与训练一致。
  - 因此带 TS 提示的两项（中文 TS-ASR 测试集、鼎桥机主识别）与交付版的差里含提示变化；普通 ASR 和热词不涉及 TS 提示。think 组与对照组完全同口径。
  - `compare` 用当前打分版本重算两边。热词两项的评测器不在 `compare` 支持范围内，只列汇总；鼎桥热词样本少（137 条），差异只作参考。
- **续训退化的初步定位**（GigaSpeech 无语音片段，共 5446 条，参考为空）
  - 有输出的条数：交付版 1515，对照组 2081，think 组 1954。
  - 对照组比交付版多出的 572 条，大多是中文短语："来来来。"50 条、"嗯。"39、"谢谢。"30、"对对对。"29。另有同一句"他主要关注的是城市中黑人及混血族群的生活。"出现 28 次。
  - 也就是说，续训后模型在英文噪声片段上更容易凭空写出中文。
  - 待查：训练数据里无语音、空参考样本的覆盖；中间 checkpoint（每 500 步一个，已在对象存储）从哪一步开始退化。

### 证据

- 训练：think 组 <https://wandb.ai/1016097967-amphion/open-audio-llm/runs/AmphionASR-spelling-think-from-v10interp050-v7set34ctxclean-3k-1node-codev1-001>，对照组 <https://wandb.ai/1016097967-amphion/open-audio-llm/runs/AmphionASR-spelling-control-from-v10interp050-v7set34ctxclean-3k-1node-codev1-001>。
- checkpoint：`whai:open-audio-llm/runs/zh-spelling-think-20261113/tasks/0{1-train-think,2-train-control}/attempts/001/artifacts/training/checkpoint-{500,…,3000}/`。
- 评测运行：`whai:open-audio-llm/runs/zh-spelling-think-20261113/ae-runs/<模型>/runs/<run>/`。

| 模型 | 上下文纠偏 think 关 / 开 | 中文 TS-ASR 测试集 | asrreg | 热词 | 警务 | 鼎桥普通 / 机主 / 热词 | CHiME-4 |
|---|---|---|---|---|---|---|---|
| `release-v10interp050`、`amphionasr2-grpo-v10interp050`（交付版） | `run-20261114T100542Z-f99aafd9` / `run-20261114T100728Z-3662588c`（`release-v10interp050-think`） | `run-20261105T182838Z-fd77aed9` | `run-20261106T021048Z-4a3e3c39` | `run-20261106T062642Z-77949f35` | `run-20261106T021103Z-24d7e316` | `run-20261106T040023Z-6d445026` / `run-20261106T040238Z-022113fb` / `run-20261106T041330Z-10e76b87` | `run-20261105T182837Z-f1fca950` |
| `control-3000` | `run-20261114T121919Z-0f1385ac` / `run-20261114T122204Z-e5b6f6ee`（`control-3000-think`） | `run-20261114T121919Z-7dd5a306` | `run-20261114T122433Z-a7fd1ba1` | `run-20261114T132125Z-596c79aa` | `run-20261114T132553Z-4ce4bd6f` | `run-20261114T132658Z-ffcb24a7` / `run-20261114T122757Z-199e98b1` / `run-20261114T132759Z-739b4064` | `run-20261114T132916Z-8d90ec37` |
| `think-3000` | `run-20261114T110535Z-510a0308` / `run-20261114T110828Z-c1c1dbb9`（`think-3000-think`） | `run-20261114T110535Z-bf5b1393` | `run-20261114T111128Z-b5ab97d1` | `run-20261114T121958Z-17c178f6` | `run-20261114T122456Z-6e7acbe2` | `run-20261114T122604Z-1032461c` / `run-20261114T111544Z-eb81a072` / `run-20261114T122713Z-0619571b` | `run-20261114T122838Z-71d83197` |

交付版上下文纠偏的两个 run 在 `ae-runs/release-v10interp050{,-think}/` 下，守门 run 在 `ae-runs/amphionasr2-grpo-v10interp050/` 下；两组 think 开的 run 在 `ae-runs/<模型>-think/` 下。

## 启动

两组各要一整个 8 卡节点（配置里是 GPU 0–7）。在本机启动前先确认 GPU 6、7 空出来了（目前给 lx 用）。

```bash
open-audio-llm experiment run --config runs/zh-spelling-think-20261113/experiment.yaml --dry-run
open-audio-llm experiment run --config runs/zh-spelling-think-20261113/experiment.yaml --task train-think
open-audio-llm experiment run --config runs/zh-spelling-think-20261113/experiment.yaml --task train-control
```

## 执行记录（2026-11-14）

### 运行位置与必要改动

- **节点**：本机 GPU 1、2、3、5 被其他任务占用，6、7 归 lx，凑不出 8 卡。用户允许用空闲 ACP 节点，且"最多占一个"。
  - 提交 1 节点 ACP 任务 `pt-58raiag3`（`AmphionASR-spelling-think-from-v10interp050-think-then-control-3k-1node`），节点 IP 10.119.17.204。
  - 任务里跑 inbox 执行器（`.local-tools/asr-holder/dev_runner.py`，目录 `/workspace/workspace/mingdong/scratch/acp-think`）。
  - 投递的 `10-train-think-then-control.sh` 依次运行 `open-audio-llm experiment run --task train-think`、`--task train-control`（代码用 `shared/code-v1`）。
  - 训练配置一字未改：节点内仍是 GPU 0–7、端口 29816。
- **W&B 转发**：配置里的 `WANDB_HTTP(S)_PROXY` 指向开发容器的 TCP 转发（10.119.17.252:17920），转发只放行登记过的节点 IP。
  - 已重启转发进程，在放行名单里加入 10.119.17.204，原有 IP 保持不变。
  - 从节点经转发访问 api.wandb.ai 返回 200。
- **交付版评测改了 GPU**：`configs/eval-release-v10interp050` 在本机跑，本机 GPU 1 被占用。
  - 用 `make_eval_configs.py release-v10interp050 <checkpoint> 0 4 18960` 重新生成，只有 TS 服务从 GPU 1 改成 GPU 4。
- **评测脚本**：
  - `make_eval_configs.py` 补写 `handoff.json`（`ae open-audio-llm after-training` 需要它）。
  - 新增 `scripts/make_gate_configs.py`：从交付版 11-05/11-06 各 run 的 `run.json` 复制守门配置，只换模型、服务地址和输出目录，因此数据、参数与交付版成绩同口径。
  - 新增 `scripts/run_eval.sh`：起两路服务后分两条通道跑评测，用到普通服务的一条，只用 TS 服务的一条。
- **评测环境**：AmphionEval main `eedcf15`（0.14.0，含 `3a5a9c1` 的 `--think`）。
  - 装在 `/workspace/workspace/mingdong/.local-tools/amphion-eval-0.14.0-eedcf15`：基于 `miniconda3/envs/amphion-eval` 建 venv，依赖相同，只装入固定提交的 amphion-eval。
  - 本机和 ACP 节点都已验证能用。
  - 0.14.0 发新 TS 提示，与两组训练用的提示一致。中文打分在 v3/v4 口径下相同（v4 只改了日语）。

### 训练启动核验（think 组 attempt 001）

- W&B：<https://wandb.ai/1016097967-amphion/open-audio-llm/runs/AmphionASR-spelling-think-from-v10interp050-v7set34ctxclean-3k-1node-codev1-001>，远端已有 `train/loss`、`train/grad_norm` 等逐步上传。
- 前 190 步 loss 在 0.13–0.23，随预热下降；grad_norm 1.2–8.4；日志无 NaN。速度约 1.1 秒/步。
- **`ce_think` 不会出现，两组都一样**：
  - 这份代码只在数据配置带 `objective` 一节时，才换成按任务记录 `ce_*` 的 retention 损失路径（`integrations/ms_swift/retention.py`）。
  - 本实验和 SFT 073 都没有 `objective`，走 swift 标准 CE，所以日志里只有 `loss`、`token_acc`。
  - 加 `objective` 会改变损失归约方式，等于改实验设计，所以没有加，也没有重启训练。
  - think 是否真正参与训练，按以下依据判断：
    - 训练时的 `data-effective.yaml` 带 think 一节，训练集据此构建 ThinkTable；
    - 预检已用真实数据核对过记录 ID；
    - 训练后 think 开时模型是否写出说明格式的 think：是，见"结果与结论"。

### 带 think 模板的服务核对（交付版，think 关）

- 两个模板渲染结果逐字相同：交付版 checkpoint 自带模板，和 `qwen3_asr_think.jinja` 在不传 `chat_template_kwargs` 时，对普通、带上下文、TS 三类消息的渲染完全一致。think 开时末尾多出 `<think>\n`。
- 用旧 TS 提示客户端（`amphion-eval-0.13.3-oldprompt`，与 11-06 口径相同）重跑：
  - 合并 CER（三个子集错误数之和 / 参考字数之和）8.85，11-06 线上为 8.88；
  - 各子集：合成 2.58（线上 2.55）、合成 TS 15.80（线上 15.89）、真实挖掘 5.62（线上 5.62）；
  - 3249 条里 3096 条输出逐字相同；
  - 配对 bootstrap（`ae open-audio-llm compare`）：差 −0.03 个点，95% CI [−0.12, +0.03]，各子集都不显著。
  - 结论：服务模板没有改变 think 关的行为，差异属于 vLLM 批处理的随机性。
  - run：`whai:open-audio-llm/runs/zh-spelling-think-20261113/ae-runs/release-v10interp050/runs/run-20261114T095648Z-3fe6d84d`。

### 训练完成与评测执行

- **训练**：两组 attempt 001 都正常结束，退出码 0。
  - think 组 09:43–10:58Z，对照组 10:58–12:14Z（UTC）。
  - W&B 两个 run 都是 finished，各 301 个 loss 点，没有 NaN 或 Inf。
  - loss 均值：前 20 个点 think 组 0.177、对照组 0.128；最后 100 个点 0.121 对 0.122。
  - 两组都没有 `ce_think`，原因见上文。
  - 各 checkpoint 由 storage_sync 传到对象存储，路径见"证据"。
- **评测**：每组最后一个 checkpoint 拉到本地后，用 `make_eval_configs.py` 和 `make_gate_configs.py` 生成配置，再用 `run_eval.sh` 跑完上下文纠偏（think 关、开）和 8 项守门。
  - think 组在本机 GPU 0、4 上，端口 18962/18963，配置在 `configs/eval-think-3000/`。
  - 对照组接在训练后，在同一 ACP 节点上（inbox 脚本 `11-eval-control-after-training.sh`），GPU 0、1，端口 18964/18965，配置在 `configs/eval-control-3000/`。
  - 所有评测退出码都是 0。
- **ACP**：对照组评测结束后，于 13:32Z 停止任务 `pt-58raiag3`，释放节点。W&B 转发放行名单里的 10.119.17.204 没有删除。

<!-- execution-status -->

执行状态：completed；质量状态：未核实。

开始：未知；结束：未结束；退出码：未知。

<!-- /execution-status -->
