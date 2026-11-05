# moss-ts-concat-20261104：注册拼接 + MOSS 说话人日志能否替代 TS-ASR

**结论：不能。** 把注册音频拼在混合音频前送 MOSS-Transcribe-Diarize、取 S01 作为目标人，在 ts-asr-benchmark@v12 全集 26,750 条上，与 10.28 版本（054avg3）逐条配对：目标在场 CER 54.3 对 13.1，漏识 31.2% 对 2.9%，目标缺席误输出 5.1% 对 3.3%，三项差值的 95% 区间都不跨 0。46 个场景组里只有 1 组 MOSS 明显更好（业务反馈 2 组样本 ID 与场景标签不对应，对比脚本单独计算，见结论文档）。

| 同一批样本、同一打分（v3） | 054avg3 | MOSS·S01 | MOSS·注册主讲人 | MOSS·S01 − 054（95% 区间） |
|---|---:|---:|---:|---|
| 目标在场 CER（12,953 条） | **13.1** | 54.3 | 53.9 | +35.0 ～ +48.8 |
| 漏识率 | **2.9%** | 31.2% | 31.5% | +27.6 ～ +29.2 |
| 目标缺席误输出（13,797 条） | **3.3%** | 5.1% | 4.4% | +1.4 ～ +2.2 |

## 失败原因

- **不是 S01 规则的问题**：注册段主讲人不是 S01 的只有 275 条（1.0%）；改用"注册区间里说话最多的人"几乎不变（CER 53.9、漏识 31.5%）。
- **是认人失败**：漏识样本里目标人的话多半转写对了，但 MOSS 给了新编号（S02/S03…），没有和注册段归为同一人。注册与混合录音条件差异越大越严重：房间响应、近远场失配、AliMeeting 近远场对照漏识 47–57%（054avg3 2–4%），电平与录音链路 53%。
- 注册含他人声时 S01 可能是混进来的人：真实会议缺席误输出 18.2% 对 1.3%。
- 复读：31 条解到 token 上限（"我我我…"），按截断内容计入分数（attempt 001 因此失败，见任务 README）。
- MOSS 唯一明显更好的组：目标缺席 · 另一个人说一句短话，误输出 2.2% 对 7.1%。真实远场会议在场 CER 接近（AISHELL-4 15.9 对 15.0），但缺席误输出 25.7% 对 5.0%。

逐组表见 [tasks/02-compare/README.md](tasks/02-compare/README.md)；阶段结论见 [docs/experiments/2026-11-05-moss-ts-concat](../../docs/experiments/2026-11-05-moss-ts-concat/README.md)。

## 做法

| 项 | 内容 |
|---|---|
| 评测集 | `open-audio-llm/ts-asr-benchmark@v12-20261022`，26,750 条，与 054avg3 运行 `run-20261028T033033Z-70075026` 同一批样本 |
| 输入 | 与 TS 服务收到的相同：注册截/补到 5 秒 + 混合音频；worker 在两者之间插 1 秒静音 |
| MOSS 推理 | 官方 vLLM（提交 68b4a1d5，`0.23.1rc1.dev949`），`/v1/audio/transcriptions`，服务端默认提示词，temperature 0，独立进程（[scripts/moss_ts_worker.py](scripts/moss_ts_worker.py)） |
| 取目标 | 开始时间在注册之后（≥ 5.5 秒）的 S01 片段；对比脚本另算"注册区间说话最多的人" |
| 打分 | AmphionEval after-training（`ae` 0.13.3，外部 JSONL worker），W&B 记录；对比脚本用同一 `tokens`/`presence_text` 对两边逐条重算（[scripts/compare.py](scripts/compare.py)） |

054avg3 当时发布的汇总用的是打分 v1（合并 CER 13.23）；本对比两边都按当前 v3 重算（13.07），口径一致。非官方用法的第三方结果，未发布到共享评测记录。

## 任务

| 任务 | 说明 | 状态 |
|---|---|---|
| [00-smoke](tasks/00-smoke/README.md) | 每来源 2 条（533 条） | 004 完成 |
| [01-eval-v12](tasks/01-eval-v12/README.md) | v12 全集 | 002 完成 |
| [02-compare](tasks/02-compare/README.md) | 与 054avg3 逐条配对对比 | 完成 |

## 证据

- W&B：全集 `run-20261105T082033Z-d4697e21`（远端 finished，1,490 项指标），冒烟 `run-20261105T065412Z-cdf0a3b6`。
- 逐条预测与 MOSS 原始输出：`tasks/01-eval-v12/attempts/002/artifacts/ae/runs/run-20261105T082033Z-d4697e21/predictions.jsonl`（`raw_text`）。
