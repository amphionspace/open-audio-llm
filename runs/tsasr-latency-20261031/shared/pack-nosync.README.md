# pack-nosync

10.28 版本交付包（054 avg3）的副本：除 `runtime/` 外均为指向原包的软链接（权重不复制）。
`runtime/vendor/qwen3_asr_vllm.py` 与 `runtime/sep_token.py` 打了与分支 `fix/tsasr-encoder-host-syncs` 相同的补丁：
长度在主机侧计算，编码器 `max_seqlen` 为 CPU 张量，`cu_seqlens` 非阻塞搬到 GPU，去掉每层一次的 GPU 同步。
