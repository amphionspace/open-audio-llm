"""Long-enrollment target-speaker ASR (AmphionASR v5) on Qwen3-ASR-1.7B.

Enrollment audio (seconds to minutes) runs through a frozen copy of encoder
layers 0-11 inside 30 s windows; each window is pooled to 32 slots, read by
8 global + 24 local queries and injected into the trainable mixture encoder
through gated cross-attention. A frozen ECAPA vector is added to the encoder
input. Plain and hotword ASR share the model with the enrollment path off.
Model code follows amphionasr-v5 (docs/guides/longenroll-v5.md).
"""
