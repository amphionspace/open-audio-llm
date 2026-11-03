# 素材音轨排查

问题：AISHELL-5 进入素材前，确认每条近讲音轨的内容和标注对得上。

- 输入：`meeting_synth_zh@v1-20261031` 的 AliMeeting、AISHELL-5 全部音轨和标注。
- 做法：[scripts/headset_leak_all.py](../../scripts/headset_leak_all.py) 按标注把每条音轨分成“只有本人说”“只有他人说”“都不说”三类时段，算功率；[scripts/select_excluded_meetings.py](../../scripts/select_excluded_meetings.py) 生成排除名单。
- 产物：`attempts/001/artifacts/`（逐音轨电平、[excluded-meetings.json](attempts/001/artifacts/excluded-meetings.json)）。
- 完成条件：两个语料全部测完，名单写入 [configs/synth-v1.yaml](../../configs/synth-v1.yaml)。已完成。

## 结果

| | AISHELL-5 | AliMeeting |
| --- | --- | --- |
| 音轨 / 会议 | 1239 / 568 | 575 / 167 |
| 串音低于本底，测不出 | 774 | 516 |
| 可测串音的中位数 | −21.3 dB | −25.1 dB |
| 串音 > −10 dB | 31 条 | 2 条 |
| 本人比本底高不到 6 dB | 22 条 | 7 条 |
| 排除的会议 | 43 场 | 6 场 |
| 损失的素材时长 | 7.6% | 3.9% |

本底取“只有他人说”和“都不说”两类时段中较低的一个，因为一些音轨的“都不说”时段里有未标注的声音（比他人时段高 10 dB 以上，来源是推断）。AISHELL-5 有 33 条音轨的时段不足 5 秒，没有测量，保留。

## 决定

- 比串音更要紧的是本人几乎不在音轨上：`P0192` 有 5 场三类时段都在 −60 dB 左右；`aishell5-train-043/P0090` 他人时段比本人时段响 6 dB；`R1019_M4056/4059` 本人时段比他人时段低 14 dB。合成器按语音电平归一化，会放大这些音轨，标注却照常保留。
- 阈值（串音 > −10 dB，或本人高出本底不到 6 dB）是看分布定的经验值，没有测过对训练的影响，代价是素材少 4–8%。
- 整场排除：2 人会议少一方就不是完整对话，用齐规则也不会再选它。只从素材排除，时间轴模板不变。

## 顺带查明：本机 NumPy 大数组分配极慢

AliMeeting 第一次测量 30 分钟没有产出，worker 用户态 1 秒、内核态约 1700 秒。原因是 NumPy 默认对大数组 `madvise` 大页，本机 `transparent_hugepage/defrag` 是 `madvise`，缺页时会同步整理内存。

| 分配 64 MB | 第 1 次 | 第 2 次 |
| --- | --- | --- |
| 默认 | 52.4 s | 71.7 s |
| `NUMPY_MADVISE_HUGEPAGE=0` | 0.03 s | 0.03 s |

设置后 AliMeeting 全量测量约 4 分钟完成，数据核查 900 窗约 9.5 分钟（attempt 004 跑了约 3 小时 40 分还没完）。对照训练配置已加这个设置；ACP 节点的内核设置未查。
