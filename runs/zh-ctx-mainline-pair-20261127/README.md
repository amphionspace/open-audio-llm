# 上下文纠偏放进主线一轮的成对对照（2026-11-27，训练前写定）

**状态**：2026-11-27 20:42Z K0 一轮跑完，判定落在“1 成立但 2 有回退”（见“结果”）；收益贴着门槛，按“风险”第一条加跑第二个种子的 K0（k0s2，任务 05–08，排队中）。

## 要回答的问题

上下文纠偏的数据和方法放进主线的一轮训练（先 SFT 再 GRPO）以后，收益有多少？会不会让其他能力回退？

之前的结论都来自“从 11.17 续训”，回退里混着“在 GRPO 模型上续训”本身的代价，分不开（`runs/zh-ctx-grpo-k-20261122` README“TS 回退的来源”）。这里用成对对照把这部分代价抵掉：两边都从 11.17 走同样的 SFT → GRPO，只差纠偏这一包。

## 依据

- 已有带纠偏的一轮：K（`runs/zh-ctx-clue-target-20261117` 第 9 问，从 11.17 起 SFT 3000 步）加上 K 的 GRPO（`runs/zh-ctx-grpo-k-20261122`，400 步，avg3，模型名 `kgrpo`）。交付评测和纠偏 v2 都已经跑过。
- 缺少的是同一轮里不带纠偏的那一边。本实验只补这一边：K0。

## 设计

K0 = K 那一轮去掉纠偏部分，其余完全照 K 的实际配置（`scripts/make_configs.py`，从 K 和 kgrpo 生成好的配置文件改，不重新生成版本）。

- **去掉的（纠偏这一包，按 K 实际训练的内容）**：
  1. 合成说字数据：`context_correction_synth_zh` 的 train、train_ts、train_ts_twins 三个来源；
  2. SFT 的 think 段（第二轮 think：说明录音和合成数据上的 think 行，加上很小比例的空 think 和普通 think），所以 K0 的每条样本都按普通识别 / TS 识别训练；
  3. GRPO 的 ctx 计划、context_targets、`qwen3_asr_context` 奖励和两路 ctx 挖题。
- **保留的**：
  - `context_correction_synth_zh` 的 train_asr_dialog。它是 K 为修长对话漏人加的普通识别长对话，不是纠偏数据。
  - 其余 131 个 SFT 来源（GRPO 133 个）的版本和权重。
  - 西语回放、热词设置、训练参数：起点 11.17、3000 步、学习率 3.55e-6、联合 MTP、种子。
  - GRPO 配方：第十轮的五份计划和份额，ctx 原来的 0.08 份额回到回放；400 步，avg3 取第 200/300/400 步。
- **代码**：SFT 用 K 的 code-v1（`oal-clue-target` 那份快照），GRPO、挖题和评测用 kgrpo 的 code-v3，都和 K 那一轮相同。
- **评测**：
  - 都和 kgrpo 同一套：纠偏 v2（think 关、开），交付 27 组（`runs/zh-ctx-grpo-20261120/scripts/acp-delivery-one.sh`，模型名 `k0grpo`）；
  - 另测 K0 SFT 本身的纠偏 v2（只记录）；
  - ae 固定用 0.14.0-eedcf15。打分口径和 kgrpo、11.17 已有的 run 相同，不混用 amphion-eval 11-27 新出的打分版本 5。

## 判定（kgrpo 对 k0grpo，配对 bootstrap 95% CI，think 关）

1. **收益**：纠偏 v2 上“需要纠正 · 明确说明写法”的真实录音（Emilia 516 条，`ctx_v2_mined_groups.py` 的分组），kgrpo 显著高于 k0grpo（CI 下限 > 0）。
   - 另记录：播客同类、独立合成题的 correct 题、语义推断题、保持率、过度纠正率。
2. **回退**：交付 27 组里，列出 kgrpo 显著差于 k0grpo 的全部项和幅度。
   - 一项只有在显著且绝对差 > 0.5 个百分点时，才记为“纠偏带来的回退”。0.5 来自 11-25 的估计：两次独立 SFT 之间的差约 0.4。
   - TS 的误触发和漏识别如果一升一降、两者之和不变差，记为门槛移动，不算回退。理由是 SFT 层面 TS 门槛在两次训练之间会移动 1–1.7 个点（`runs/zh-ctx-clean-pair-20261125` 后续结果）。
3. **结论规则**：
   - 1 成立且 2 没有回退：纠偏这一包在主线一轮里有净收益，可以进主线；
   - 1 成立但 2 有回退：把收益和回退的全表交给用户，由用户取舍；
   - 1 不成立：在一轮 SFT → GRPO 里，纠偏这一包拿不到明确说明类的收益，不进主线。
- **只记录**：两边对 11.17 的全表；K 对 K0 在 SFT 层面的纠偏 v2。

## 风险

- **只有一对**：运行间波动会混进差值，判定 2 的 0.5 个点门槛和 TS 门槛移动规则就是为此设的。差值落在临界附近时，再跑一个不同种子的 K0 来估波动，写在结果里。
- **K 是 11-22 跑的**：代码快照、数据版本、配置都用 K 当时的。集群和节点不同，不影响确定性以外的结果。
- **“CE 只训正文”这一条方法不在内**：K 的 SFT 训练器（ms-swift）没有按来源屏蔽标签前缀的开关。这里测的是 K 实际训练的那一包。
- **K0 去掉 think 以后，think 开的数字没有意义**：think 开的结果只对 K 记录。

## 执行

- ACP pt-f4eiiqi0（1 节点 8 卡），11-27 14:01Z 提交，14:02Z 分到节点 10.119.16.252（W&B 转发已放行）。节点脚本 14:11Z 放入，K0 SFT 随即开始。执行器 `/workspace/workspace/mingdong/scratch/acp-ctx-mainline-pair`，节点脚本 `scripts/acp-node.sh`，跑完由 `stop_when_done.sh` 停节点。
- 预计：SFT 约 1.5 小时，挖题约 1 小时，GRPO 约 40 分钟，评测约 2.5 小时，合计约 6 小时。

## 结果（2026-11-27，详见 `tasks/04-eval/README.md`、`tasks/04-eval/gate.md`）

**判定 1 成立，判定 2 有回退 → 按规则把全表交给用户取舍。**

- **收益**：Emilia 需要纠正 · 明确说明（516）45.54 → 47.67，+2.13 [+0.19, +4.07]；其中训练没见过的 469 条 +2.35 [+0.21, +4.48]。合成题 correct 率 +13 到 +32 个点，过度纠正不显著。
- **回退**（纠偏这一包带来的，显著且 > 0.5）：西语 TS 误触发 v1/v2 4.4 → 9.3 / 4.3 → 9.0（出在 K 的 SFT，全部不在场场景一起翻倍）；chime4 WER 8.97 → 11.39（出在 K 的 GRPO，漏识别）；ts-hotwords-es CER +1.45；ts-en 误触发 +1.31；scene-ts CER +1.06；first-other CER +0.65；ts-hotwords CER +0.51。
- **同时变好**：ctx-v1 CER 6.50 → 1.31，crosslingual-en / es WER -2.22 / -4.48，asr-benchmark 英文误触发 -1.78。
- **新发现**：对 11.17 的真实录音纠偏收益大部分已经在 K0 里（没见过的明确说明：11.17 40.30 → K0 45.84 → K 48.19）。K0 保留了 finals 里的 ctxclean 改正文本和 5 个带说明录音子集的上采样（按普通识别训），去掉的合成数据 + think + GRPO 纠偏只多约 2.4 个点。
- **第二个种子**：收益 CI 下限只有 +0.19，按“风险”第一条加跑 k0s2（`scripts/make_configs.py` ARMS：训练种子 43、数据种子 20261128、挖题 epoch +100，其余同 K0；任务 05-sft-k0s2、06-mining-k0s2、07-grpo-k0s2、评测名 k0s2grpo）。节点 pt-bije92x2（A 线让出，`/workspace/workspace/mingdong/scratch/acp-v99e` 的 `ctx-k0s2.sh`），等 TS 线 t29 守门约 22:45Z 结束、8 张卡全空后启动，约 6 小时。kgrpo 对 k0s2grpo 用同一个 gate.py 再算一遍；两个 K0 之间的差就是运行间波动。

<!-- execution-status -->

执行状态：completed；质量状态：未核实。

开始：未知；结束：未结束；退出码：未知。

<!-- /execution-status -->
