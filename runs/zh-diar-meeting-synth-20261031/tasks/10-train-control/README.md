# 对照训练

问题：第二轮 I 组配方加在线重配音会议，能否降低真实会议 cpER。

- 输入：[configs/train-control.yaml](../../configs/train-control.yaml)、[configs/train-data-control.yaml](../../configs/train-data-control.yaml)，都由 [scripts/make_control_configs.py](../../scripts/make_control_configs.py) 生成；起点 Emilia2 6000。
- 配比：在线合成会议占 24.7%（30 s 15.9 / 120 s 5.5 / 300 s 3.3），权重来自 `sot_multispeaker_zh_en` 的多人部分、`sot_many_speaker_word_zh_en` 的 75%、`emilia2_zh_dialogue_diarize` 的 50%。素材 AliMeeting 和 AISHELL-5 各半。
- 完成条件：训练结束，W&B 远端有指标，选点由 [../11-select-control/README.md](../11-select-control/README.md) 判定。

## 进展

未提交。用户要求等 `pt-m70602xw`、`pt-6cri11hg` 都结束，其中 `pt-6cri11hg` 已结束。提交前已通过：dry-run、全配方预检（`tracking/preflight-control-local.json`）、[数据核查 005](../03-data-check/README.md)。

提交时用 `env -u PYTHONPATH`，否则任务名会多出 `ASR-` 前缀。

2026-11-02 改动（用户：“将main merge过来确保以后提pr没问题”）：
- 本分支合并 origin/main（提交 1fea5cb，全量测试 483 passed / 6 skipped），代码快照改为 `shared/code-control-v2`；旧快照 `shared/code-control` 留作首次预检的记录。
- 补上 W&B 转发：ACP 访问 api.wandb.ai 被拒，I 组训练的跟踪就以“远端结束状态未核实”结束。配置现在设 `WANDB_HTTP(S)_PROXY: http://10.119.16.21:17922`。**提交前先在开发机启动转发**：`scratch/grpo/wandb_forward.py 10.119.16.21 17922 17911 <pod IP>`，先放行宿主 IP 和 10.119.17.1-254，pod 起来后从 /proc/net/tcp 读到 pod IP 再收窄（见 acp-dev-runner 记录）。
- 新快照在本机 GPU 0 上做了小试，见 [../15-control-smoke/README.md](../15-control-smoke/README.md)。
- `pt-m70602xw` 是 GRPO 工作流的开发机任务（inbox 脚本持续跑评测），不会自行结束；等它“结束”需要那条工作流或用户释放节点。

2026-11-03 再次合并 main（a7d8eb2，含 #109、#99、#110），合并提交 e8f5d6b，无冲突，全量测试 492 passed / 1 skipped，代码快照改为 `shared/code-control-v3`，train/select dry-run 通过（`tracking/dry-run-*-003-code-v3.log`）：
- #109 改了默认行为：训练音频超过满幅时按峰值缩回 1.0 再写 FLOAT WAV，和部署时的 PCM16 输入一致；在线合成会议也走这一步。`level_prob` 默认关，本配方不开。
- #99 的功能提交 938e72e 本分支早已包含（I 组同款语速扰动），这次只多了合并修正。
- #110 让数据预检多出 `levels` 段（只报告，不设通过线），训练任务的预检会自动带上。
- 本配方没有纯 ASR 空目标（会议空目标记录都被 `exclude_records` 排除），所以另一会话指出的“空目标偏安静”问题不涉及本训练。
- 用 v3 快照重跑全配方预检，通过（`tracking/preflight-control-local-v3.json`），1000 个样本中空目标为 0。按 `levels.by_source` 统计的响度 loud_dbfs p50：合成会议 −31.7 / −32.6 / −31.9 dBFS（30 / 120 / 300 s），真实远场会议 AliMeeting −27 ~ −37、AISHELL-5 −27 ~ −32、AISHELL-4 −31 ~ −34，合成会议的电平落在真实远场范围内；近讲的 Emilia2 对话是 −15.2。合成会议在样本中占 246/1000，配方是 24.7%。
