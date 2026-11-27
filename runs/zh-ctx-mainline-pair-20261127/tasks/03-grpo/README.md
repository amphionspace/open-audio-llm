# 03 K0 GRPO（400 步）

- **输入**：`configs/train-grpo.yaml`（kgrpo 的配置，起点换成 K0，去掉 `qwen3_asr_context` 奖励）和 02 的五份计划。
- **完成条件**：400 步跑完，checkpoint-200/300/400 上传；W&B 远端 finished、有指标。
- **进展**：
  - attempt 001（17:22–18:02Z，同一节点）：完成，400 步，退出码 0。
  - 生效配置核对过：奖励是 accuracy、presence、hotword、format 四项，没有 `qwen3_asr_context`；数据配置里没有 ctx 计划和 context_targets；起点和 MTP 初始化都是 K0 的 checkpoint-3000。
  - W&B 远端 finished，global_step 400，最后 train/reward 1.69（https://wandb.ai/1016097967-amphion/open-audio-llm/runs/AmphionASR-ctx-mainline-pair-k0-grpo-v10recipe-noctxplan-from-k0-3000-textfinal-mtp-8gpu-400steps-001）。
  - 平均每步约 6 秒；评测阶段 18:03Z 开始。

<!-- execution-status -->

执行状态：completed；质量状态：未核实。

开始：2026-11-27T17:22:09.258724+00:00；结束：2026-11-27T18:02:15.906080+00:00；退出码：0。

对象存储：已同步到 `whai:open-audio-llm/runs/zh-ctx-mainline-pair-20261127/tasks/03-grpo/attempts/001`。

<!-- /execution-status -->
