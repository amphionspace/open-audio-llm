# 01 守门拆分

## 做什么

原来的中文 TS 守门只看字错率、误报、漏识三个合计数，过去的轮次几乎都在误报和漏识之间互换。这个任务在不改守门判定的前提下，给每次评测加几项拆分指标。

## 工具

`../../scripts/ts_gate_breakdown.py`：

- 读一次评测的 `predictions.jsonl` 和 `samples.jsonl`。
- 用 AmphionEval 的分词和归一化（用 amphion-eval 环境的解释器运行），输出以下指标：
  - **出不出字**：误报条数、整条漏识条数，以及两者在全部字错里的占比。
  - **转写错误**：目标在场且输出了文字的样本上 (S+D+I)/参考字数，按 `target_overlap_ratio` 分成 0 / 0–0.6 / 0.6–0.9 / ≥0.9 四档。
  - **同句干扰代价**（`--asr-run`）：同一模型在普通识别测试集上单独识别同一原句（按语料和句子 id 对上）的错误率，以及 TS 场景比它高多少。
- `--wandb` 把这些数作为 `breakdown/*` 写进这次评测自己的 W&B run（run id 就是 AmphionEval 的 run id）。

## 完成条件

- 在 11.17 上复现复盘报告的数字。
- W&B 远端能读回这些指标。

## 结果

在 11.17 上的复现（v12 run `run-20261117T032925Z-961b82b3`，普通识别 run `run-20261117T042922Z-7f90ddc5`）：

| 指标 | 值 |
|---|---|
| 误报 / 漏识 | 342 / 357 |
| 出不出字在全部字错里的占比 | 29.2% |
| 转写错误：重叠 0 / 0–0.6 / 0.6–0.9 / ≥0.9 | 6.11 / 9.44 / 10.29 / 15.64 |
| 同句干扰代价 | 3.10 → 13.86（+10.76），11,760 条 |

结果与报告一致。W&B `run-20261117T032925Z-961b82b3` 远端读回 29 个 `breakdown/*` 指标，原来的 1,485 个评测指标没有变。

## 用法

```bash
/workspace/workspace/mingdong/miniconda3/envs/amphion-eval/bin/python scripts/ts_gate_breakdown.py \
  --ts-run <v12 run> [--asr-run <同模型普通识别 run>] --out breakdown.json --wandb
```
