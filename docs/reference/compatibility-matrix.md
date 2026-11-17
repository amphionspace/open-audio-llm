# Compatibility Matrix

| Area | Default | Optional / Legacy |
| --- | --- | --- |
| Python | 3.10+ | 3.11 recommended for ms-swift |
| Model API | Hugging Face `PreTrainedModel` | legacy `.pt` conversion |
| LLM | `AutoModelForCausalLM` with `inputs_embeds` | unsupported LMs need adapters |
| Audio encoders | Qwen3-ASR, Qwen3-Omni adapters | Zipformer legacy adapter |
| Training | ms-swift SFT/GRPO | legacy `src/train.py` during migration |
| Serving | vLLM 0.32 serving profile (CUDA 12.9) | HF `generate` fallback; GRPO in-process rollout stays on vLLM 0.18 |
| k2 | not required | legacy Zipformer training only |
