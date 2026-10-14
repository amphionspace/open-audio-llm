# 权重插值回原版（不训练）

问题：第一次学说话人转写（unified-8000）让单人识别比 Qwen3-ASR 原版差 3–9 点，之后各轮只收回一部分；AliMeeting Eval 干净单人远场语句上原版 5.0、6000 步 7.4（farfield-gap-diagnosis-20261013）。

做法：`a × 原版 + (1 − a) × 6000 步`，a = 0.1 / 0.2 / 0.3 / 0.5，`sep_token` 保留 6000 步的值，原版独有的 `lm_head` 与其词嵌入相同、跳过。与 6000 步本身（a = 0）在同一次评测里比较：

- 选模面板：AliMeeting dev cpER，另离线算与他人重叠 / 无重叠语句的错误率；
- 单人 ASR dev：WenetSpeech、KeSpeech 各 1000 条，官方请求；
- AliMeeting Eval 单人语句远场与近讲（1,099 句）。

判读：若某个 a 让单人指标明显回升，而多人指标（重叠语句错误率、cpER）不变差，就把它作为下一轮的起点或部署候选。

## 任务

- [01-merge](tasks/01-merge/)：插值。
- [02-eval](tasks/02-eval/)：评测，GPU 6。
