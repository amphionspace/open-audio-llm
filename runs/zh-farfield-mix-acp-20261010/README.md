# 中文远场配比：ACP 16 卡全参数

用户决定（2026-10-10）：老师复查后的中文会议标注与中文远场数据就绪后，停止 [中文优先配比](../zh-first-mix-acp-20261010/README.md)（6000 步），从其最优 checkpoint 继续。只换中文数据，配比总量不变：

| 组 | 占比 | 数据 |
| --- | --- | --- |
| AISHELL-4 | 10 + 10 | v2 修正标注：8 路均值 / 单个原始麦克风 |
| AliMeeting | 8 + 8 + 4 | v2 均值 / 单麦 / 近讲模拟远场 |
| RAMC | 5 + 3 | v2（老师补标）/ 模拟远场 |
| 英文会议 | 6 | AMI 3、NOTSOFAR 2、CHiME-6 1（不变） |
| 合成 | 28 | 词级中文 15、英文 3、旧合成 10（不变） |
| 单人回放 | 12 + 6 | 原回放按比例缩到 12；中文远场回放 6（RealMAN、3D-Speaker clean、ViTW） |

采样种子 45；学习率 5e-6、500 步 warmup 后 cosine 至 24000 步，新优化器，2×8 A800，每 2000 步保存并在开发机 GPU 7 选模。

## 为什么做

续训与中文优先两轮的真实会议指标都在约 6000 步后进入平台期；与 MOSS 的中文差距主要是远场替换错误、误检和说话人混淆。老师复查显示人工标注基本完整（漏标 <0.03 h），主要问题是 AliMeeting 36 场时间偏移（已排除）；此前训练只用 8 路均值音频，与真实单麦远场输入不一致。本轮检验修正标注 + 单麦/模拟远场数据对中文远场的收益。

## 数据要点

- 标注修正、两处切窗/合并缺陷修复与重建：`/workspace/workspace/mingdong/data/amphiondata/real-meetings-20261010/README.md`。
- 排除清单 `zh-v2-train-excluded-records.txt`：zh245 选模集 18 场留出会议 + 空目标窗口（与 v1 训练一致）。
- 单麦视角需要 `catalog_dataset` 按 `AudioRef.channel` 读取混合音频（`ab56564`），本轮代码快照包含该修复。
- 3D-Speaker 有 clean 版本，按 `require_clean_pass: true` 使用；RealMAN、ViTW 无 clean 版本，使用现有训练集。

## 判定

与之前相同：只在排除复读的 AliMeeting dev cpER 连续 2 次高于基线（8000 合并模型，22.10%）+ 1 点时停止；复读条数不超过 17 + 6 的 checkpoint 才作为候选。结束评测 meeting-180s（重点看中文真实会议）、CHiME-6 dev、固定 338 条集，与起点和 MOSS 对照。执行完成与质量达标分开记录。

## 任务

- [01-prepare](tasks/01-prepare/README.md)：数据配置（执行 002；001 为脚本权重校验错误，未产出）。
- [02-train](tasks/02-train/README.md)：ACP 训练。
- [03-select](tasks/03-select/README.md)：开发机选模，停止文件 `tasks/03-select/control/stop-request.json`。
