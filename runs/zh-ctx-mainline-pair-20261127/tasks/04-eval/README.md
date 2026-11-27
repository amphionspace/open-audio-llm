# 04 评测与判定

- **输入**：k0grpo-avg3（03 第 200/300/400 步平均，ae 缓存 model-1fb861ad4779375d7dc02418）、start-k0（01 的 checkpoint-3000，缓存 checkpoint-3000-29899967e537）；对照 kgrpo-avg3、11.17 已有的 run。
- **做法**：`scripts/eval_after_training.sh`：纠偏 v2（think 关、开），交付 27 组（模型名 k0grpo）；比较用 `runs/zh-ctx-grpo-20261120/scripts/compare_delivery.py k0grpo kgrpo`（ts-asr 组走 `ae open-audio-llm compare`）和 `scripts/gate.py`。
- **完成条件**：全部退出码 0，W&B 远端 finished；按 README“判定”逐项给出结论。
- **进展**：完成。节点脚本 `31-k0-mining` 18:03Z 进入评测，20:42Z 结束（k0grpo-avg3=0 start-k0=0 delivery-k0grpo=0），节点 pt-f4eiiqi0 20:43Z 由 stop_when_done 停掉。
- **W&B**：本次评测的 31 个 ae run（交付 27 + 纠偏 v2 的 k0grpo-avg3 / start-k0 各 think 关、开）逐个在远端核过，全部 finished 且有指标。

## 判定结果（`gate.md`、`gate.json`，kgrpo − k0grpo，think 关，配对 bootstrap 95% CI）

**结论规则落在“1 成立但 2 有回退”**：收益和回退的全表交给用户取舍。

### 1. 收益

| 项目 | k0grpo | kgrpo | 差 [95% CI] |
|---|---:|---:|---|
| **Emilia 需要纠正 · 明确说明（516，判定项）** | 45.54 | 47.67 | **+2.13 [+0.19, +4.07]** 成立 |
| Emilia 需要纠正 · 语义推断（629） | 68.68 | 69.16 | +0.48 [-0.96, +2.07] |
| Emilia 原转写已对 · 明确说明（662，保持） | 90.94 | 91.84 | +0.91 [-0.45, +2.27] |
| 播客 需要纠正 · 明确说明（45） | 53.33 | 60.00 | +6.67 [+0.00, +15.56] |
| 合成 ASR correct · 明确说明（395） | 48.61 | 68.86 | +20.25 [+15.95, +24.56] |
| 合成 TS correct · 明确说明（378） | 38.89 | 70.90 | +32.01 [+26.72, +37.30] |
| 独立合成 ASR correct · 明确说明（1000） | 28.00 | 41.00 | +13.00 [+10.50, +15.50] |
| 独立合成 TS correct · 明确说明（784） | 18.62 | 37.24 | +18.62 [+15.56, +21.56] |
| 合成 / 独立合成 过度纠正（confirm 题 target_alternative） | — | — | 都不显著（-3.0 到 +1.7） |

按训练是否见过拆开（`train_overlap`，只记录；1,680 条没见过、223 条见过）：没见过的“需要纠正 · 明确说明”469 条 +2.35 [+0.21, +4.48]，没见过的全部 1,680 条 +1.19 [+0.24, +2.14]，和判定项一致，不是记忆。

### 2. 回退（显著且差 > 0.5 个点）

| 项目 · 指标 | k0grpo | kgrpo | 差 [95% CI] | 出在哪一段 |
|---|---:|---:|---|---|
| ts-es-v1 误触发 | 4.40 | 9.26 | +4.86 [+4.31, +5.38] | K 的 SFT（11.17 3.21 → K 8.78 → kgrpo 9.26） |
| ts-es-v2 误触发 | 4.33 | 9.03 | +4.70 [+4.25, +5.16] | 同上 |
| chime4 WER / 漏识别 | 8.97 / 5.53 | 11.39 / 8.03 | +2.42 / +2.50 | K 的 GRPO（K 6.54 → kgrpo 11.39） |
| ts-hotwords-es CER | 25.66 | 27.11 | +1.45 [+0.88, +2.00] | 召回反而 +1.30* |
| ts-en 误触发 | 6.30 | 7.61 | +1.31 [+0.97, +1.63] | |
| scene-ts CER | 3.92 | 4.99 | +1.06 [+0.29, +1.90] | |
| first-other CER | 19.97 | 20.62 | +0.65 [+0.38, +0.94] | 误触发 -2.46*、漏识别 +1.68*（门槛移动） |
| ts-hotwords CER | 14.06 | 14.57 | +0.51 [+0.23, +0.81] | 刚过 0.5 |

- 门槛移动（不算回退）：first-other（漏 +1.68 / 误 -2.46）、crosslingual-es（误 +2.50 / 漏 -4.86）。
- 显著但 ≤ 0.5（只记录）：v12 CER +0.35、v12 漏 +0.29、babble3/4 CER +0.24、ts-en WER +0.25 等。
- 显著更好：ctx-v1 CER 6.50 → 1.31、crosslingual-en WER -2.22、crosslingual-es WER -4.48、asr-benchmark 英文误触发 -1.78、babble4 误触发 -1.10、v12 误触发 -0.49、scene-asr 漏 -0.39。

**西语 TS 误触发的形态**（`ts-es-v1`，目标说话人不在场的 6,480 条）：k0grpo 285 条有输出，kgrpo 600 条；涨幅摊在全部约 50 个不在场场景上（每个场景大约翻倍，纯非语音场景都是 0），输出是干扰人的西语原话。这是“在场判断整体往输出那边偏”，不是个别难例。

**竖线输出**（参考里没有，只记录）：k0grpo / kgrpo = hotword-bench 126 / 54、ctx-v1 0 / 3。两边都带 train_asr_dialog（159 条带竖线），所以 K0 也有。

## 只记录：两边对 11.17

纠偏 v2（think 关；`ctx_v2_mined_groups.py` 分组，Emilia 按 train_overlap 拆）：

| 项目 | 11.17 | start-k0（K0 SFT） | k0grpo | kgrpo |
|---|---:|---:|---:|---:|
| Emilia 需要纠正 · 明确说明（516） | 39.15 | 45.93（+6.78*） | 45.54（+6.40*） | 47.67（+8.53*） |
| 其中没见过的（469） | 40.30 | 46.48（+6.18*） | 45.84（+5.54*） | 48.19（+7.89*） |
| 没见过的全部（1,680） | 67.20 | 70.60（+3.39*） | 69.94（+2.74*） | 71.13（+3.93*） |
| 合成 ASR correct · 明确说明（395） | 6.33 | 49.11 | 48.61 | 68.86 |

K0 去掉的是合成说字数据、think 和 GRPO 的纠偏计划与奖励；保留的有 finals 里的 ctxclean 改正文本、5 个带说明录音子集的上采样（按普通识别训，约 1.1% 份额）和 train_asr_dialog。**对 11.17 的纠偏收益大部分（没见过的明确说明 +5.5 / +7.9）已经在 K0 里**，去掉的那一包只多 +2.4。

交付 27 组对 11.17（同一条回退规则：显著且 > 0.5；`compare/r1117-vs-k0grpo/`、`r1117-vs-kgrpo/`，热词组按 gate.py 自助抽样）：

| | 11.17 → k0grpo | 11.17 → kgrpo |
|---|---|---|
| 回退 | v12 误触发 +1.49、first-other 误触发 +2.19、scene-ts 误触发 +10.0（21.4 → 31.4，70 条里多 7 条）、ts-es v1/v2 误触发 +1.19 / +1.30、crosslingual 误触发 +0.80、crosslingual-es 漏 +1.67 | ts-es v1/v2 误触发 +6.05 / +6.00、ts-en 误触发 +1.42、scene-ts CER +1.55、chime4 WER +1.96 / 漏 +2.80、ts-hotwords-es CER +0.78、first-other CER +0.51 |
| 门槛移动 | 无 | v12、first-other、crosslingual-en、crosslingual-es |
| 显著更好（摘要） | v12 CER -0.27 / 漏 -0.91、scene-asr CER -2.49、ts-hotwords 召回 +3.71、ts-es v1/v2 WER -1.0 / -1.1、ctx-v1 -1.37、babble-es -0.70 | scene-asr -2.70、crosslingual -3.69、ctx-v1 -6.56、ts-es v1/v2 WER -0.9 / -1.1、crosslingual-es -3.02 |

K0 对 11.17 的回退全是 TS 误触发往上走、漏识别往下走（v12 两项之和 +0.58，刚过门槛），和“在 GRPO 模型 11.17 上再做一轮 SFT”的已知代价同一形态（`runs/zh-ctx-grpo-k-20261122` README“TS 回退的来源”）。本实验没有不带真实数据部分的对照，分不开它和 ctxclean / 说明录音上采样各占多少，这里只记录。

## 证据

- `gate.md` / `gate.json`（`scripts/gate.py`，约 49 s）；`runs/zh-ctx-grpo-20261120/tasks/04-delivery/compare/k0grpo-vs-kgrpo/` 和 `delivery-k0grpo-vs-kgrpo.md`。
- ae run 根目录：`/workspace/data/ae-runs/ctx-mainline-pair-20261127-{k0grpo-avg3,k0grpo-avg3-think,start-k0,start-k0-think}`、`ctx-grpo-20261120-delivery-k0grpo`。
