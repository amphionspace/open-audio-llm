代码快照：任务 11（循环编码器）。在 code-v2（feat/ts-clean-teacher-kl 023232e，任务 07/08 的代码）上打了 feat/ts-audio-loop 的补丁：新增 `src/open_audio_llm/tsasr/audio_loop.py`，并让 HF 逐条前向（sep_token.py）、训练批处理前向（audio_batching.py）、模型加载（register_qwen3_asr.py）和仓库 vLLM 后端（vllm_backend.py）按 `amphion_audio_loop` 配置插入循环层。和 08 的代码只差这一处。
audio-data-contract：与 code-v2 相同。
评测服务用同目录的 `shared/pack-loop`（pack-mtp 运行时加同样的循环补丁）。
