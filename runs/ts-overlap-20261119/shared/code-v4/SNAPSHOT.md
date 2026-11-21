代码快照：任务 12（从 11.17 起的 GRPO，加重叠转写计划）。复制自 `runs/zh-ctx-grpo-20261120/shared/code-v3`（worktree oal-ctx-grpo）：open-audio-llm main 15eacbf（#171 挖题 TS 片段补齐 Mel 帧、#172 rollout 的 embeds 处理器接受 vLLM 0.18 关键字调用，都已合入），上下文纠偏 GRPO 用它跑通了挖题、训练和评测，训练文本为终版。
audio-data-contract-src 与那边相同（audio-data-contract 4aaa3df）。
没有循环编码器补丁（11.17 不需要）。
