# 说字写法 think 第一轮（2026-11-13，已准备、未启动）

**状态**：配置和数据都已就绪，训练没有启动（用户 2026-11-13："你先做好，然后训练先不启动"）。

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

## 启动

两组各要一整个 8 卡节点（配置里是 GPU 0–7）。在本机启动前先确认 GPU 6、7 空出来了（目前给 lx 用）。

```bash
open-audio-llm experiment run --config runs/zh-spelling-think-20261113/experiment.yaml --dry-run
open-audio-llm experiment run --config runs/zh-spelling-think-20261113/experiment.yaml --task train-think
open-audio-llm experiment run --config runs/zh-spelling-think-20261113/experiment.yaml --task train-control
```
