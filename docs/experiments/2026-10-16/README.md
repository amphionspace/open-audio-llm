# 2026-10-10 至 10-15：中文会议说话人日志（Diarization）训练系列总结

**结论：这一系列没有产出可替换的新模型，起点仍是 Emilia2 轮 6000 步（`zh-emilia2-dialogue-acp-20261012` select 004）。** 7 轮训练里，只有 Emilia2 轮产出了现在的起点；此后各轮在中文真实会议（AliMeeting dev 不计复读 cpER）上都没有稳定超过它；与 MOSS 的差距（我们 20.2、MOSS 15.2）没有缩小。按 2026-10-16 的决定，Diarization 训练到此收敛，算力转给 TS-ASR。

## 各轮结果

| 轮次（`runs/`） | 做了什么 | 结果 |
| --- | --- | --- |
| zh-first-mix / zh-farfield-mix / zh-dialogue-mix（10-10） | 三种配比：中文优先、加远场、加公开中文对话 | zh-first-mix 6800 步、zh-farfield-mix 约 20000 步停止，AliMeeting dev 在 20.0–21.6 内波动（平台期），结论是调配比无效，需要更多真实中文对话数据；zh-dialogue-mix 的 README 没有记录最终结果 |
| zh-asr-retention（10-12） | 单人 ASR 保持 | 约 4540 步按用户要求停止；目标 WenetSpeech 反而变差，原因未核实 |
| zh-emilia2-dialogue（10-12） | Emilia2 1–3 人对话伪标签 | 选 6000 步作起点；只改善合成多人，不改变中文真实会议 |
| farfield-gap-diagnosis（10-13） | 诊断远场差距 | 远场声学、微调损伤、窗口长度都不是主因；差距在重叠句和说话人 |
| zh-overlap-farsim（10-13） | Emilia2 重叠远场重排（20%） | 不采用；重叠句字错没有改善，最终面板未过 |
| weight-interp（10-14） | 10% 原版权重插值 | 选模面板更好（20.34），最终面板变差（合成中文 10.2 → 18.1），不采用 |
| zh-similar-voice（10-14） | 真实会议和 Emilia2 里音色相近的窗口加权 | 不采用；高相似归错 3.10 → 3.5–5.8（均高于起点），详见[运行目录](../../../runs/zh-similar-voice-acp-20261014/README.md) |

## 沉淀的结论

1. **与 MOSS 的差距主要在说话人归属，不在字**：AliMeeting dev 5.0 点差距里约 2.1 点是归错，且随声纹相似度上升（低/中/高相似窗口差距 0.9 / 2.2 / 3.9）。
2. **MOSS 伪标签会把一个人拆成多个说话人**，尤其有声书和广播剧。用它的标注训练，模型学会把相近的声音拆开。人工标注的 AliMeeting 里相似度超过 0.60 的最近两人只占 1%，Emilia2 相似子集占 29%。听审：0.6–0.8 段多数是真两个人，≥ 0.8 段多数是同一人，拆人上限应约 0.80（样本小）。
3. **仿真重叠不迁移**：合成多人一直占约 15%，合成会议上我们大幅领先 MOSS（10.0 对 17.5），真实会议却落后；再加 20% 重叠仿真，重叠句字错没有变化。
4. **选模面板的胜出不可信，必须看最终面板**：10% 权重插值在选模面板最优，最终面板的合成、远场、车载都明显变差。
5. **单次评测有约 1 点噪声**：同一个 6000 步模型测得 20.63 / 21.06 / 21.08 / 21.52。小于 1 点的差异不能当真，每轮都要在同一口径重测起点。
6. **对外比较只用 120 秒窗口上模型自己的输出**，不引入跨窗口关联。

## 对 TS-ASR 有用的部分

- 做"音色相近"负样本或注册语音污染时，不要直接用 MOSS 标注判断"是否另一个人"；先用声纹相似度（ERes2NetV2）和人工标注的分布核对，再定阈值。
- 相似度脚本与听审流程：`/workspace/workspace/mingdong/data/amphiondata/similar-voice-views-20261014/scripts/`（`emilia2_split_cap.py`、`make_split_review.py`），听审页用 `amphiondata review build/serve`。
- 选模器 `runs/zh-similar-voice-acp-20261014/scripts/select_checkpoints.py` 可直接复用：异步评测每个保存点、基线重测、门槛与提前停止、`previous_state` 续跑。
- 共享 GPU 的坑：vLLM 启动要求空闲显存 ≥ `gpu-memory-utilization` × 总量；启动前看 `nvidia-smi`，避开 GPU 0；失败后改 `gpus` 并用 `previous_state` 续跑，不要从头来。已有两轮因此失败（GPU 2、GPU 3 被他人占用）。
- 配置里不要引用会被删除的工作树路径（overlap 轮的 `chime6-dev.yaml` 因此在最终面板失败）。

## 数据与产物位置

| 内容 | 位置 |
| --- | --- |
| 音色相近视图（已注册 `similar-voice-v1-20261014`） | `/workspace/workspace/mingdong/data/amphiondata/similar-voice-views-20261014/`（README 有登记与拆人上限记录） |
| 与 MOSS 的逐窗对比、声纹相似度 | `/workspace/workspace/mingdong/data/model-compare-review-20261014/` |
| 各轮选模历史、最终面板 | 各 `runs/<轮>/tasks/03-select/attempts/*/artifacts/` |
| W&B | entity `1016097967-amphion`，project `open-audio-llm`，run 名前缀 `zh-similar-voice-acp-20261014` 等 |

## 没做的事

- 没有用 0.80 上限重算 Emilia2 排除名单，也没有用清洗后的数据重训，所以"拆错的伪标签导致归错变多"只是有证据的解释，没有被重训证实。
- 4 人及以上窗口（本地 0.9B 只有 51–74% 可用）用 MOSS Pro 重打的工作没有收尾，不确定是否完成。

## 复盘与调研

见 [strategy-review.md](strategy-review.md)：结论是 7 轮都在同一配方上调配比，已到平台期；外部做法的差别在数据量与标签质量、解码结构、说话人任务的单独训练。恢复前先做不占 GPU 的三件事（声纹重聚类修标签、核对说话人损失、把 TS-ASR 与日志合成一条线）。

## 如果以后重启

先用 0.80 上限重算排除名单，再决定要不要重训；起点用 Emilia2 6000 步，选模看高相似窗口归错和 cpER，最终面板必须通过后才算采用。
