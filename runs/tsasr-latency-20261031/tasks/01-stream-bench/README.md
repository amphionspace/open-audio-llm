# 01 流式回放测延迟与吞吐

- 输入：v12 真实会议条目（AISHELL-4/AliMeeting/Emilia2/RAMC，在场与缺席），10.28 版本交付包。
- 方法：`scripts/stream_bench.py` 逐个起服务变体，按 audiollm-go main 的节奏模拟并发会话（1/16/48 路，各 300 s），另做 max_tokens=1 的 prefill 微基准（线上指令 vs 空指令）。
- 完成条件：每个变体输出首字、文字更新间隔、断句到 final 的 p50/p95，请求级 token 与延迟，服务端 prefill/decode 时间与投机接受率。
- 进展：见 attempts。
