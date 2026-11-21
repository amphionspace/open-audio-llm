open-audio-llm main 4ca977e（#171 合入之后）。和 code-v1（main f044443）只差 `src/open_audio_llm/scripts/grpo_mining.py`：TS 挖题的注册音频和混音按训练的方式补齐到整数个 Mel 帧（`prepare_ts_clips`）。code-v1 的 TS 挖题请求因此全部被 TS 服务以 400 拒绝（2026-11-21 第一次在 ACP 上挖题）。
audio-data-contract-src 从 code-v1 复制，仍是 audio-data-contract 4aaa3df（main 的 requirements-ci.txt 固定这个版本）。
