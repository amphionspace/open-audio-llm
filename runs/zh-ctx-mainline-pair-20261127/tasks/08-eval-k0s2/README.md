# 08 第二个种子的评测与判定

- **为什么**：README“风险”第一条。K0 一轮的收益 CI 下限只有 +0.19，用第二个种子的 K0（k0s2grpo）估运行间波动。
- **输入**：k0s2grpo-avg3（07 第 200/300/400 步平均）、start-k0s2（05 的 checkpoint-3000）；对照 kgrpo、k0grpo、start-k0 已有的 run。
- **做法**：
  - 节点上 `scripts/eval_after_training.sh k0s2`（同 04：纠偏 v2 think 关 / 开，交付 27 组，模型名 k0s2grpo）。
  - 开发机上 `runs/zh-ctx-grpo-20261120/scripts/compare_delivery.py k0s2grpo kgrpo` 和 `compare_delivery.py k0grpo k0s2grpo`，然后 `scripts/gate_seed2.py`。它用 gate.py 同一套规则算三对，结果写到本目录：
    - `k0s2grpo-vs-kgrpo/`：判定再算一遍；
    - `k0grpo-vs-k0s2grpo/`：两边都不带纠偏，只看波动；
    - `start-k0-vs-start-k0s2/`：两个 K0 SFT 的纠偏 v2，只记录。
- **完成条件**：节点脚本退出码 0，评测 run 在 W&B 远端 finished；三对结果写出，结论写进本 README 和实验 README。
- **进展**：完成。节点脚本 `ctx-k0s2-retry.sh` 11-28 04:03:48Z 结束，退出码 0（k0s2grpo-avg3=0、start-k0s2=0、delivery-k0s2grpo=0）。本次 31 个评测 run（交付 27 + 纠偏 v2 两个模型各 think 关 / 开）和 GRPO run（`…-400steps-seed2-001`，400 步）逐个在 W&B 远端核过，全部 finished 且有指标。gate.py 加了 `OUT` / `TITLE`，列名改用 `MODELS`；默认输出和已提交的 `tasks/04-eval/gate.md`、`gate.json` 逐字节相同（重跑核对过）。

## 结果（think 关，配对 bootstrap 95% CI，同一条回退规则：显著且 > 0.5 个点）

### 收益：换一个 K0 基线仍然成立，K0 之间的波动比收益小

| Emilia 组（条数） | k0grpo | k0s2grpo | kgrpo | kgrpo − k0grpo | kgrpo − k0s2grpo | k0s2grpo − k0grpo |
|---|---:|---:|---:|---|---|---|
| **需要纠正 · 明确说明（516，判定项）** | 45.54 | 45.16 | 47.67 | +2.13 [+0.19, +4.07] | **+2.52 [+0.39, +4.65]** | -0.39 [-2.13, +1.36] |
| 需要纠正 · 语义推断（629） | 68.68 | 68.52 | 69.16 | +0.48 | +0.64 | -0.16 |
| 原转写已对 · 明确说明（662，保持） | 90.94 | 91.24 | 91.84 | +0.91 | +0.60 | +0.30 |
| 播客 需要纠正 · 明确说明（45） | 53.33 | 51.11 | 60.00 | +6.67 | +8.89* | -2.22 |

- 两个 K0 SFT（start-k0 / start-k0s2）之间同样不显著：明确说明 45.93 / 44.96（-0.97 [-2.71, +0.78]）。
- k0s2grpo 对 11.17：明确说明 39.15 → 45.16（+6.01*），保持 88.07 → 91.24（+3.17*），和 k0grpo（+6.40*、+2.87*）一致。**“对 11.17 的收益大部分已经在 K0 里”这一条在第二个种子上复现。**

### 回退：哪些是纠偏这一包的，哪些和运行间波动一样大

| 项目 · 指标 | kgrpo − k0grpo | kgrpo − k0s2grpo | k0s2grpo − k0grpo（波动） | 归类 |
|---|---|---|---|---|
| ts-es-v1 误触发 | +4.86* | +4.72* | +0.14 | **稳定回退** |
| ts-es-v2 误触发 | +4.70* | +4.76* | -0.06 | **稳定回退** |
| ts-hotwords-es CER | +1.45* | +2.00* | -0.56* | **稳定回退** |
| ts-en 误触发 | +1.31* | +2.24* | -0.92* | **稳定回退**（幅度 1.3–2.2） |
| first-other CER | +0.65* | +0.82* | -0.16 | **稳定回退**（误触发 / 漏识别本身是门槛移动） |
| ts-hotwords CER | +0.51* | +0.48* | +0.03 | 贴着 0.5，两次都显著 |
| customer-hotwords CER | +1.54 | +2.22* | -0.68 | 方向一致，一次显著 |
| chime4 WER / 漏识别 | +2.42* / +2.50* | +1.10 / +0.61 | +1.32* / +1.89* | 和波动同量级，不能算到这一包上 |
| scene-ts CER | +1.06* | -0.32 | +1.39* | 波动 |
| babble4 误触发 | -1.10* | +0.68* | -1.78* | 波动 |
| asr-benchmark 英文误触发 | -1.78* | +1.14* | -2.92* | 波动 |
| badcase CER | +2.22 | +6.67* | -4.44 | 波动（小集合，CI 宽） |
| crosslingual-en / es | 门槛移动 | 门槛移动 | 不显著 | 门槛移动 |

- **两个 K0 之间按同一条规则也会“回退”**：scene-ts CER +1.39*、chime4 WER +1.32* / 漏识别 +1.89*。所以 0.5 个点的统一门槛对这两组太紧，主线门槛要给它们单独的余量（至少按这次实测的波动 1.4–1.9 个点）。
- **同时变好的**（两个基线都显著）：ctx-v1 CER（-5.19 / -3.75）、crosslingual-es WER（-4.48 / -4.23）、crosslingual-en WER（-2.22 / -1.27）。
- **竖线输出**（只记录）：hotword-bench k0grpo 126、k0s2grpo 39、kgrpo 54；ctx-v1 0 / 0 / 3。两个 K0 差 3 倍，竖线条数本身也随运行波动。

### 结论

- 判定 1 在第二个基线上仍成立，且两个 K0 之间的差（-0.39）远小于收益（+2.1 到 +2.5）。纠偏这一包在真实录音上的净收益是真的，大约 +2.3 个点。
- 稳定的代价集中在 TS：西语 TS 误触发翻倍（+4.7 到 +4.9），英文 TS 误触发 +1.3 到 +2.2，西语 / 中文 TS 热词 CER 变差，first-other CER +0.7 到 +0.8。chime4 和 scene-ts 原来记的回退和运行间波动一样大，撤回。
- 按 README 的结论规则仍然是“1 成立但 2 有回退”。按用户 11-28 的安排，这一包不单独上线，取舍写进并入主线交接单（实验 README“并入主线交接单”第 4 项）。

## 证据

- `k0s2grpo-vs-kgrpo/gate.md`、`gate.json`；`k0grpo-vs-k0s2grpo/gate.md`、`gate.json`；`start-k0-vs-start-k0s2/ctx.json`（`scripts/gate_seed2.py`）。
- 交付比较：`runs/zh-ctx-grpo-20261120/tasks/04-delivery/compare/{k0s2grpo-vs-kgrpo,k0grpo-vs-k0s2grpo}/` 和 `delivery-*.md`（`compare_delivery.py`）。
- ae run 根目录：`/workspace/data/ae-runs/ctx-mainline-pair-20261127-{k0s2grpo-avg3,k0s2grpo-avg3-think,start-k0s2,start-k0s2-think}`、`ctx-grpo-20261120-delivery-k0s2grpo`。
