# 03-serve / 03-serve-moss：评测用 vLLM 服务

- 03-serve：起点 8000 步合并模型，vLLM 0.18.0，GPU 1，端口 8761，served name `step8000`；BF16、TRITON_ATTN、eager、seed 42、max_model_len 12288、并发 16、编码器整段注意力（n_window_infer 26000）。
- 03-serve-moss：MOSS-Transcribe-Diarize，官方固定提交 vLLM `0.23.1rc1.dev949+g68b4a1d58`（env moss-vllm），GPU 3，端口 8762。执行 001 在 GPU 2 启动时因其他进程释放显存导致显存探测断言失败；执行 002 改用 GPU 3。
- 两个服务在评测完成后由 SIGTERM 停止，状态记为 interrupted，属预期。评测快照 `server-runtime.json` 记录 `/version` 与实际加载的权重路径。
