# max_overlap 0 的复查（作废，未解码）

attempt 001 启动约 2 分钟后手动中止（exit 143）：发现变体没有保住时间轴（见 ../04-fidelity-variants-clean），结果无判定意义。

## 为什么撤回 max_overlap 0

> 2026-11-01 补充：撤回 max_overlap 0 仍然成立，但下面"不成立"的说法过强：多报说话人的比例确实随来源会议人数和窗口外的人的串音秒数上升，修法是用齐来源会议的所有人。见 [../06-split-probe/README.md](../06-split-probe/README.md)。

原假设是素材里未标注的原对话他人（串音）造成重配音偏难、模型多数说话人。用现有 v1 解码核对后不成立：

- 多出来的说话人说的是什么（[extra-speakers.json](../02-fidelity-check/attempts/002/artifacts/extra-speakers.json)，
  [scripts/extra_speaker_analysis.py](../../scripts/extra_speaker_analysis.py)）：近讲 87.5% / 远场 64.9% 是标注里已有的话，
  原对话他人的话只有 0.1% / 3.5%（随机对照 0.2% / 0.6%）。即模型把一个人拆成两个，不是在转写串音。多出的说话人只占输出的 1.5% / 2.3%。
- 文字错误与串音秒数没有剂量关系（[fill-leak-seconds.json](../02-fidelity-check/attempts/002/artifacts/fill-leak-seconds.json)）：
  近讲 0 s 3.0%，0.5–1 s 7.1%，≥5 s 8.0%，更像"被重叠过的语句本身更难"。
- 多数说话人在近讲与远场几乎相同（+0.41 / +0.43），而 −22 dB 的串音在远场多被噪声盖住。
