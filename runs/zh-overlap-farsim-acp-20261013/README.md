# Emilia2 重叠远场重排：ACP 单机 8 卡全参数

依据 [farfield-gap-diagnosis-20261013](../farfield-gap-diagnosis-20261013/)：AliMeeting dev 上与 MOSS 约 4 点的文字差距中，约 3.4 点来自与他人重叠的语句（我们 24.8%，MOSS 20.2%），远场声学、微调损伤和窗口长度都不是主因。训练数据里能学重叠的真实数据只有 AliMeeting（约 100 h 原始音频，平均重叠率 34%），Emilia2 对话几乎无重叠（1.4%）。

本轮加入 `emilia2_zh_overlap_farsim@overlap-farsim-v1-20261013`：Emilia2 2–3 人真实对话逐句重排为重叠远场窗口，74,971 窗、1,535 h，重叠率均值 0.23，SLR28 真实房间 + MUSAN 噪声；文字与说话人为 MOSS 0.9B 伪标签。试跑难度：6000 步 / MOSS 不计复读 cpER 34.4 / 25.6。数据说明见 amphiondata MR !6 与运行目录 README。

## 配比

| 组 | 占比 | 内容 |
| --- | --- | --- |
| Emilia2 轮全部来源 | 80 | 对话轮 64 + Emilia2 对话 16，内部比例不变 |
| 重叠远场重排 | 20 | 不再加噪声（数据已含噪声和混响） |

## 训练

1 节点 8 卡，每卡 2 条、梯度累积 2（全局 batch 32），学习率 5e-6、500 步 warmup 后 cosine 至 12000 步，每 1000 步保存；起点为 Emilia2 轮选出的 6000 步（整场会议 AliMeeting Test 23.7、AISHELL-4 40.8），新优化器。

## 判定

- 目标：AliMeeting dev 不计复读 cpER 最低。
- 门槛与 Emilia2 轮相同：6–12 人合成不升 5 点、WenetSpeech/KeSpeech dev 各不升 1 点、复读不超过起点 + 6。
- 只报告不设门槛：AliMeeting dev 与他人重叠语句（>3 字）的文字错误率（起点 23.7%）、无重叠语句的错误率（8.4%）、跨说话人重复输出的窗口数（1）。
- 结束后跑最终面板，并补测 benchmark-zh-20261011 三层（重点：整场会议 cpER）。

## 结果（2026-10-14）

结论：不采用。重叠语句没有改善，最终面板未通过；当前最好的仍是起点 Emilia2 6000 步。

- 训练：6280 步时被选模叫停（6000 步连续保持退化：6–12 人合成 30.5，门槛为起点 25.2 + 5）。
- 选模：AliMeeting dev 不计复读 cpER 21.52 → 2000 步 20.11（最低）；与他人重叠语句 23.69 → 23.95，无重叠语句 8.97 → 7.59。
- 最终面板（2000 步 vs 起点，select 004；起点结果取 Emilia2 轮 select 004 的最终面板）：

| 面板 | 起点 | 2000 步 |
| --- | --- | --- |
| AISHELL-4 test 180 s | 23.5 | 22.0 |
| AliMeeting test 180 s（不计复读） | 21.1 | 22.6（21.1，复读 0 → 1） |
| 合成中文 / 英文 / 中英 | 10.2 / 12.6 / 11.3 | 13.4 / 13.2 / 13.6 |
| AISHELL-5 dev（不计复读） | 45.7（27.8） | 49.3（27.7） |
| CS-Dialogue test（不计复读） | 11.8（7.2） | 9.8（7.5） |
| fixed338 AISHELL-4 / AliMeeting | 18.4 / 20.8 | 18.1 / 20.4 |
| CHiME-6 dev（仅报告） | 62.5 | 73.4 |

- 复核（`/workspace/workspace/mingdong/data/model-compare-review-20261014`，与 MOSS 的逐窗对比页及脚本）：按说话人对齐（与 cpER 同口径）后，
  AliMeeting dev 重叠语句错误为起点 22.4、2000 步 21.8、MOSS 16.1；其中纯文字部分 18.7 / 18.6 / 15.1，说话人归属部分 3.7 / 3.1 / 1.0。
  重叠仿真没有改善重叠语句的文字识别；按时间拼接的旧口径会把插话打断处的断句顺序差异记成错误，低估了与 MOSS 的差距（3.7 → 5.7 点）。
- 执行：train 002 完成；select 003 因 `configs/data/chime6-dev.yaml` 仍指向已删除的 catalog 工作树而在最终面板失败，004 沿用 003 的选模历史只补跑最终面板。

## 任务

- [01-prepare](tasks/01-prepare/)：数据配置。
- [02-train](tasks/02-train/)：ACP 训练。
- [03-select](tasks/03-select/)：开发机选模，停止文件 `tasks/03-select/control/stop-request.json`。
