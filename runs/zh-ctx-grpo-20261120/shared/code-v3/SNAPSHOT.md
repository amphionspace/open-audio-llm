open-audio-llm main 15eacbf（#172 合入之后）。和 code-v2（main 4ca977e）只差 `src/open_audio_llm/integrations/vllm/plugin/qwen3_asr_embeds.py`：GRPO 训练的 rollout（vLLM 0.18）按关键字调用 `_call_hf_processor`，#165 只认位置参数，于是 code-v2 的 GRPO 每个 rank 第一步就报错（2026-11-21）。
audio-data-contract-src 从 code-v1 复制，仍是 audio-data-contract 4aaa3df（main 的 requirements-ci.txt 固定这个版本）。
