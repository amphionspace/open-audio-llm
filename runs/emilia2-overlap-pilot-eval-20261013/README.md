# Emilia2 重叠远场重排试跑：难度核对（不训练）

数据：amphiondata `emilia2_zh_overlap_farsim` 试跑 3（`/workspace/workspace/mingdong/data/amphiondata/emilia2-overlap-farsim-20261013/pilot`，
223 个窗口、4.7 h，平均重叠率 0.22），未登记，方法见该目录 README。评测链与 farfield-gap-diagnosis-20261013 相同（vLLM 服务 + AmphionEval meeting，GPU 3）。

## 结果

| 模型 | cpER | cpER（不计复读） | 说话人无关 CER | 复读 | DER | 人数正确率 |
| --- | --- | --- | --- | --- | --- | --- |
| zh-emilia2-6000 | 62.55 | 34.43 | 33.76 | 7 | 0.209 | 0.69 |
| moss-transcribe-diarize | 36.53 | 25.63 | 26.24 | 2 | 0.122 | 0.77 |

对照 AliMeeting dev（选模面板）：我们 20.8 / 21.6，MOSS 15.6 / 18.5。

- 比真实会议更难，两边都高约 8–14 点；差距（约 8.8 点）比 AliMeeting（约 5 点）大，参考文本来自 MOSS 0.9B，MOSS 有天然优势。
- 不是退化数据：重叠段长度分布与真实会议接近（中位 / p90 / p99：试跑 0.7 / 2.1 / 7.9 秒，AliMeeting 训练集 0.8 / 3.3 / 7.0 秒，AISHELL-4 0.6 / 1.8 / 5.5 秒）。
- 我们模型的错误：被重叠的长句 42.4%（AliMeeting 24.8%），无重叠长句 15.5%（AliMeeting 7.9%；含伪标签误差与加噪）。
- 7 个窗口出现复读（AliMeeting dev 为 0）：训练时这类数据要和复读监测一起看。

## 任务

- [01-eval](tasks/01-eval/)：两模型在试跑窗口上的评测。W&B：`emilia2-overlap-pilot-eval-20261013-eval-001`。
