# ts-overlap-20261119：重叠时的目标转写

## 背景

2026-11-19 的复盘和白盒分析（`docs/experiments/2026-11-19-ts-retro-overlap/`，PR #167）发现：

- 11.17 的中文 TS 字错里，71% 是目标在说话、模型也出了字，但重叠时转写错了。
- 10 个版本在重叠 ≥ 0.9 样本上的转写错误率一直在 15.4–16.5%，任何一轮都没有改变它。
- 用户 2026-11-20 同意按复盘的建议推进：先改守门（方向 1）、测注册的影响（方向 3），同时准备干净原句老师的训练代码（方向 2）。
- 中英体验不回退优先；中文 TS 守门沿用 11.06 的 12.88 / 371 / 369。

## 任务

| 任务 | 内容 | 状态 |
|---|---|---|
| [01 守门拆分](tasks/01-gate-breakdown/README.md) | `scripts/ts_gate_breakdown.py`：在原守门之外，报出"出不出字"和按重叠分箱的转写错误，以及同句干扰代价，写回该评测自己的 W&B run | 完成 |
| [02 注册长度](tasks/02-enroll-length/README.md) | 11.17 在 v12 上分别用 3/5/8/10/12 秒注册（只评测） | 完成 |
| [03 干净原句 KL](tasks/03-clean-kl/README.md) | 从 11.17 起，v96 配方，解冻音频塔，加干净原句老师的 token 级 KL | 未达标：重叠转写比对照组还差 |
| [04 对照](tasks/04-control/README.md) | 同 03，不加 KL | 未达标：从 11.17 接着训，全面退化 |
| [05 老师信号诊断](tasks/05-teacher-signal/README.md) | 在训练分布样本上量老师和学生对标注的把握（11.17 / 03 / 04） | 完成：老师≈标注，KL 集中在听不到的字上 |
| [06 v12 参考打分](tasks/06-v12-refscore/README.md) | 两组在 v12 参考文本上 teacher forcing，配对比较 11.17 | 完成：对照组拟合变差，实验组解码漏字；编码器选择性都没变 |
| [07 音频塔表示损失](tasks/07-clean-feature/README.md) | 从 11.17 起，冻结 LLM，只训音频塔，混音上的音频 token 拉向起点音频塔在干净原句上的 token | 冒烟 |
| [08 冻结 LLM 对照](tasks/08-frozen-llm/README.md) | 同 07，不加表示损失 | 冒烟 |

## 代码

- `shared/code-v1`：任务 03/04，分支 `feat/ts-clean-teacher-kl` 318c9bb 的 src/ 和 audio-data-contract，见 [SNAPSHOT.md](shared/code-v1/SNAPSHOT.md)。
- `shared/code-v2`：任务 07/08，023232e，加了音频塔的表示损失，见 [SNAPSHOT.md](shared/code-v2/SNAPSHOT.md)。
