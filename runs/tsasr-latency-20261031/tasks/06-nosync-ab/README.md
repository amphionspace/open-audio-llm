# 06 编码器去 GPU 同步 A/B

- 背景：05 剖析显示每个请求的音频编码有约 30 次 GPU→CPU 同步（编码器每层一次 `max_seqlen.item()`，以及 `sep_token.py` 里长度的 `.tolist()`/`.item()`），在分时共享的 GPU 上每次都要等其他进程的时间片。
- 输入：原交付包 vs `shared/pack-nosync`（同权重，runtime 打 `fix/tsasr-encoder-host-syncs` 的补丁）。
- 方法：原版/补丁版交替各两轮；每轮先做 200 条单路完整解码比对原始输出，再测 prefill（1/16 路）与流式（16/48 路）。
- 完成条件：输出逐字一致（或差异不超过同版本两轮之间），并给出延迟变化。
