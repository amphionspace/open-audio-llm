# 重叠语音白盒分析（2026-11-19）

报告与结论见 `docs/experiments/2026-11-19-ts-retro-overlap/README.md`。本目录只放脚本和执行记录，不训练模型。

## 任务

| 任务 | 输入 | 完成条件 | 状态 |
|---|---|---|---|
| 01 字错拆分（CPU） | 各模型 v12 `samples.jsonl` / `predictions.jsonl` | 误报 / 漏识 / 替换 / 漏字 / 多字和按条件分箱的表 | 完成，见报告第二节 |
| 02 同句对照（CPU） | 11.17 的 v12 run 与普通识别 run | 同一原句单独识别 vs TS 场景的字错率 | 完成，见报告第二节 |
| 03 参考文本逐字打分 + 换注册（GPU） | 11.17 checkpoint（ae 模型缓存 `whitebox-merge-v18a-es116inc`）、v12 run `run-20261117T032925Z-961b82b3` 的全部目标在场样本 | 8 个分片全部 exit 0；HF 贪心与 vLLM 输出逐 token 一致率 ≥ 0.99 | 完成：8 分片 exit 0（12,953 条），一致率 0.9952；W&B `1deqonfe` 已远端核验 |

## 脚本

- `scripts/decompose_errors.py`：字错拆分、抄干扰人比例（干扰人原文二元组命中，随机样本作对照）。
- `scripts/clean_vs_ts.py`：按语料与句子 id 对上普通识别测试集输出。
- `scripts/ref_score.py`：输入构造与 `runs/whitebox-20261107/scripts/extract_states.py` 相同（ms-swift 模板、5 秒注册、音频塔全局注意力）。每条样本跑三遍前向：自己的输出（核对与 vLLM 一致）、参考文本、换成别人的注册。
- `scripts/analyze_ref_score.py`：汇总 03。
- `scripts/wandb_log.py`：把汇总写进 W&B（只写汇总，不写逐条样本）。

## 执行记录

- 03：ACP 任务 `AmphionASR-whitebox-overlap-refscore-encswap-1117-v12-1node`（`pt-wmi0k75o`），逐条输出与汇总存档在 `whai:open-audio-llm/runs/whitebox-20261119-overlap/tasks/03-ref-score/`，代码用 `open-audio-llm-ts-robust/runs/ts-robust-acp-20261015/shared/code-v37/src`（与 11.17 训练同一份）。
