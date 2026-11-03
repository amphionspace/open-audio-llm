# 在线会议合成器与保真度检查（2026-10-31）

方案第 1–2 步（诊断与方案见 [../zh-diar-sim-real-gap-20261031](../zh-diar-sim-real-gap-20261031/README.md)）：
离线建素材库，训练时在线合成会议。真实会议时间轴（谁在何时说、压着谁说）作模板，每轮用另一场真实对话
连续的近讲语音重新配音，再经真实房间冲激响应、各人音量差和会议室噪声。代码在分支
`feat/online-meeting-synthesis`（`src/open_audio_llm/data/meeting_online.py`，训练入口为 `catalog_dataset`
的 `online_meeting` 来源）。训练前先在 AliMeeting dev 138 个选模窗口上做三项保真度检查，自动判定，不设人工听审。

## 任务

| 任务 | 状态 | 结果 / 证据 |
| --- | --- | --- |
| build-pool | 完成 | `meeting_synth_zh@v1-20261031`：模板 AliMeeting 167 / AISHELL-4 176 / AISHELL-5 568 场；素材 26.1 万句（AliMeeting 110.7 h、AISHELL-5 93.9 h）；zh245 留出的 18 场不进池。[tracking/build-meeting-pool.log](tracking/build-meeting-pool.log) |
| synth-stats | 完成，通过 | 合成窗口的对话统计贴近真实：AliMeeting 模板的嵌入式插话 5.11 次/分（真实 4.4，旧合成器约 0）。[tracking/synth-label-stats-120s.json](tracking/synth-label-stats-120s.json) |
| fit-acoustics | 完成（第 4 次） | 见下节。[tracking/far-acoustics-train.json](tracking/far-acoustics-train.json) |
| fidelity-variants | 完成（attempt 002） | 001 发现并修复噪声数字静音导致的全零样本（训练同样受影响，已在合成器中修复）。[tasks/01-fidelity-variants](tasks/01-fidelity-variants/README.md) |
| fidelity-check | 完成 | 声学通过（各重叠档与真实远场一致）；重配音偏难，主要多在说话人归属。多报说话人随来源会议人数增加，用齐来源会议后大幅下降（见 whole-conversations）；机制未定，可能是残留未标注串音，也可能是缺席者造成的对话不连贯。重配音素材都是起点模型见过的语音，结果偏乐观。[tasks/02-fidelity-check](tasks/02-fidelity-check/README.md) |
| data-check | 完成（attempt 005，最终配置） | 900 窗走训练路径，泄漏和标注通过；退出码 1 来自 1 窗 3 帧数字静音，判定不影响训练。未标注串音的训练影响无法事先测出。[tasks/03-data-check](tasks/03-data-check/README.md) |
| split-probe / far-gap-probe | 完成 | 多报随来源会议人数变化（2 人来源 13.5%，4 人来源 45.4%），不是背题也不是声音一致性问题；远场文字差距主要来自上下文被打断。[tasks/06](tasks/06-split-probe/README.md)、[tasks/07](tasks/07-far-gap-probe/README.md) |
| whole-conversations | 完成，采用 | AliMeeting 素材用齐来源会议：近讲 cpER 18.9–19.8（原 22.4–23.4，真实 16.35），多报 2 人窗 13–16%（原约 62%）。[tasks/08](tasks/08-whole-conversations/README.md) |
| material-audit | 完成，采用 | 全部音轨测串音和本人电平，排除 49 场（AISHELL-5 43、AliMeeting 6）。[tasks/09](tasks/09-material-audit/README.md) |
| aishell5-material | 完成，保持 1:1 | 起点模型在 AISHELL-5 素材上明显更差（远场不含循环 41–43，3 人窗少报 42%），未发现标注缺陷；影响由选点判定。[tasks/12](tasks/12-aishell5-material/README.md) |
| aishell5-listening | 完成，保持 1:1 | 12 题 A/B 盲听：合并的 8 对答不同 4、同一 1、听不出 3；同一人对照 2/2，不同人对照 1/2。未发现标注两人实为一人的证据。[tasks/13-aishell5-listening/README.md](tasks/13-aishell5-listening/README.md) |
| control-smoke | 完成，通过 | 合并 main 后的快照在单卡上训练 40 步：loss 有限、数据加载不拖慢、W&B 远端有指标。[tasks/15-control-smoke/README.md](tasks/15-control-smoke/README.md) |
| train-control / select-control | 未提交 | 11-03 再次合并 main（#109/#99/#110），配置改用快照 `code-control-v3`，预检通过；已补上 ACP 的 W&B 转发。`pt-m70602xw` 是 GRPO 的开发机任务，不会自行结束。[tasks/10](tasks/10-train-control/README.md)、[tasks/11](tasks/11-select-control/README.md) |
| fidelity-variants-clean / fidelity-check-clean | 作废 | `max_overlap: 0` 破坏时间轴且依据不成立，已撤回。[说明](tasks/05-fidelity-check-clean/README.md) |

## 声学参数（fit-acoustics）

训练集远场（阵列各通道均值，与训练、评测窗口一致）AliMeeting、AISHELL-4 各 60 场：

| | 底噪信噪比 p10–p90 | 语音电平 p10–p90 | 各人电平差 std |
| --- | --- | --- | --- |
| AliMeeting | 17.6–27.0 dB（中位 22.3） | −43.5 至 −28.7 dBFS | 2.19 dB |
| AISHELL-4 | 6.8–15.8 dB（中位 12.0） | −41.7 至 −35.1 dBFS | 2.36 dB |
| 合并 | 8.0–26.2 dB | −42.8 至 −30.4 dBFS | 2.29 dB |

- 噪声取录音的底（100 ms 帧功率第 5 百分位，排除数字静音帧）。前三次失败：001/002 用标注空隙当噪声，
  空隙里有未标注的说话声，噪声高估 15–25 dB；003 遇到 5% 以上为全零的录音，噪声底为 0 而除零崩溃。
  第 4 次的噪声底与噪声库单通道底噪中位差 −2 dB（阵列平均略低），可信。
- 合成器按噪声片段平均功率配比信噪比，噪声库片段平均功率比自身底高 4.4 dB，所以 `snr_db` 取测量值减 4.4。
  `level_db_std` 由测得的电平差按每场约 4.1 人、±2σ 截断换算为 3.0。推导写在
  [configs/synth-v1.yaml](configs/synth-v1.yaml) 注释里。
- 训练用合并范围（[configs/synth-v1.yaml](configs/synth-v1.yaml)）；保真度检查的对照是 AliMeeting dev，
  用 AliMeeting 范围（[configs/synth-v1-fidelity-alimeeting.yaml](configs/synth-v1-fidelity-alimeeting.yaml)）。

## 下一步

`pt-m70602xw` 结束后提交对照训练（round 2 arm I 配方加在线合成会议 24.7%，AliMeeting 与 AISHELL-5 素材各半），按平滑配对规则判定。
