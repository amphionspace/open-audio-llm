# 只改配比：真实会话 45% + 词级 6–12 人合成（ACP 16 卡）

上一轮（`many-speaker-full-acp-20261008`）把真实中文会议从起点训练的 40% 降到 20%，新增 50% 整句朗读合成。结果是多人合成大幅改善（180 秒 6–10 人 85.88% → 17.13%），但 AliMeeting 与 AMI 出现会话应答词复读（180 秒复读 12 → 26 条），中文选模第 1000、1500 步连续超出容差而停止。新训练数据中没有复读目标；同起点、原配比的全参续训第 500 步也退化 1.35 点，配比变化使其放大。

本轮只验证这一根因：**只改数据配比与合成版本**，起点、学习率、调度、批大小、步数全部与上一轮相同。

## 配方（样本比例）

| 组 | 比例 | 来源 |
| --- | ---: | --- |
| AISHELL-4 | 12% | `meeting-long-v1-20260920` 官方 train，沿用上一轮四种窗口与排除 |
| AliMeeting | 12% | 同上 |
| AMI | 6% | `ami_sdm_meeting_replay@v1-20261008` 官方 train（上一轮 2%） |
| NOTSOFAR | 6% | `notsofar_sdm_meeting_sot@real-meetings-v1-20261008` train，按会议去重均衡 |
| CHiME-6 | 4% | `chime6_meeting_sot@real-meetings-v1-20261008` train（U01 四通道均值） |
| RAMC | 5% | `magicdata_ramc_meeting_sot@real-meetings-v1-20261008` train，真实双人对话 |
| 词级 6–12 人合成（中） | 17% | `sot_many_speaker_word_zh_en@word-v1-20261008`，7 个人数等权 |
| 词级 6–12 人合成（英） | 8% | 同上 |
| 旧 1–5 人合成 | 15% | 上一轮来源与内部比例（含 1 条 matched A/B 事件源，权重极小） |
| 单人 ASR 回放 | 15% | 上一轮来源、说话人排除与 clean 实际通过过滤 |

真实会话合计 45%（会议 40% + RAMC 5%）。上一轮整句 6–12 人合成 `sot_many_speaker_zh` 不再使用。数据集内部四种窗口（约 30/120/300/600 秒）按记录数分配，与上一轮相同。保留 `epoch_samples: full_coverage`。

- **词级合成分组**：Catalog 只有一个 train 划分，用 `exclude_records` 拆成语言 × 人数 14 个来源，每个人数等权。
- **NOTSOFAR 去重**：同一会议有 4 或 5 台设备（14 场 4 台、58 场 5 台），按设备数分区，权重 ∝ 记录数 / 设备数，使每场会议按去重后的窗口贡献，而不是按设备数放大。
- 新数据已带混响与噪声（合成）或是真实远场录音，不再叠加训练噪声。
- 数据准备只组装配置，不生成音频：`tasks/01-prepare`（执行 002）。

## 训练

起点 `unified-diarization-asr-20261001` 8000 步合并模型；全新优化器、调度器与采样游标。全参数更新，LLM/aligner 1e-5、编码器 5e-6，500 步 warmup 后 cosine 至 2000 步；每卡 2 条、全局 32 条；每 500 步保存。2 节点 × 8 A800-80GB（ACP）。

训练回调只做契约核对（全部参数可训练、16 ranks、full_coverage、学习率首次检查为正）和停止请求读取。停止文件在 `tasks/03-select/control/stop-request.json`：ACP 容器以 root 写训练目录，开发机不可写，所以由开发机在此目录写入，rank 0 每 5 步读取并广播。提交前该目录必须存在、文件必须不存在。保存后 rank 0 给 checkpoint 加读权限（`ReadableCheckpointCallback`，已在代码快照）。

## 选模与结束评测（开发机，部署推理栈）

- 每个保存点复制权重到选模执行（本地已被上传回调删除时从对象存储拉回，只拉权重），`open-audio-llm serve` 起 vLLM HTTP 服务，由 AmphionEval `ae open-audio-llm meeting` 评测。两张卡并行评相邻 checkpoint，不用 GPU 0。
- 选模面板 `meeting_selection_panel@v1-20261008`：AliMeeting 官方 dev（中文保持）与 NOTSOFAR dev；目标为 6–12 人合成 70 条（AISHELL 官方 dev 说话人）。
- 起点基线为同一新链路测得（`amphion-eval-selection-20261008` 06-baseline-select 执行 001，复制在 `tracking/selection-baseline-step8000.json`）：AliMeeting dev 22.12%、NOTSOFAR dev 125.77%、6–12 人 112.51%、复读 17 条。不再沿用旧脚本的 13.66%（旧口径在折叠复读后打分）。
- **停止**：AliMeeting dev 连续 2 次比基线差超过 1 个百分点，或面板复读条数连续 2 次超过基线 17 条（增量上限 0，沿用该分支模板默认值）。
- **候选**：中文与复读都满足的 checkpoint 中 6–12 人 cpER 最低者；没有合格候选时，用最后评测的 checkpoint 出结束报告，选模判为未通过。
- **结束评测**：meeting-180s、CHiME-6 dev（仅报告）、固定 338 条集，全部新链路，与起点逐组比较。meeting-180s 与 CHiME-6 基线已测（`tracking/final-baseline-step8000-*.json`）；固定集无新链路基线，结束时在起点模型上补测。固定集由 `tasks/04-export-fixed` 导出为本地数据（不是共享登记）。

## 质量条件

执行完成与质量达标分开记录。达标需同时满足：AliMeeting dev 不超容差、复读不增加、180 秒中文真实会议与英文真实会议不劣于起点（容差 1 点）、6–12 人合成改善。

## 版本边界与已知限制

- AmphionEval 的 `meeting` 评测命令与选模面板评测集尚未发布（MR !18、audio-data-contract PR #20 为草稿），评测从 `../amphion-eval-meeting/src` 分支源码运行，`AUDIO_DATA_CATALOG` 指向 `../audio-data-contract-panel`。合并发布后应改为固定版本。
- 固定 338 条集的 AliMeeting 138 条与选模面板的 AliMeeting dev 窗口相同，只有 AISHELL-4 test 200 条独立于选模。
- NOTSOFAR dev 12 名说话人中 10 名也在 train；英文面板对说话人不独立。
- 复读增量上限 0 在新链路重跑时可能因个别样本复读翻转而误触发；需连续 2 次才停止。
- 词级合成短应答每句平均被复用约 33 次；英文合成没有真实应答。
- aidatatang 源为全量 600 人（含其官方 dev/test 说话人），只用于合成训练，未用于评测。

## 任务与证据

- [01-prepare](tasks/01-prepare/README.md)：组装数据配置与子集 ID 文件。
- [02-train](tasks/02-train/README.md)：ACP 训练。
- [03-select](tasks/03-select/README.md)：开发机选模与结束评测。
- [04-export-fixed](tasks/04-export-fixed/README.md)：固定集导出。
- `tracking/`：代码与依赖快照、CPU 数据预检、基线副本。

正式入口：`open-audio-llm train --config runs/many-speaker-real-mix-acp-20261009/configs/train.yaml`；选模 `open-audio-llm prepare --config runs/many-speaker-real-mix-acp-20261009/configs/select.yaml`。
