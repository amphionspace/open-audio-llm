"""On-policy GRPO for target-speaker ASR (Qwen3-ASR + learned SEP).

``prepare`` draws on-the-fly TS samples from a Catalog data config, ``train``
runs GRPO with HF generation on the policy itself, and ``evaluate`` scores a
checkpoint greedily through the TS-ASR vLLM service. See
docs/guides/tsasr-grpo.md.
"""
