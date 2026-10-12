# 01-decode-runs：解码设置对照与默认解码重跑

## 输入

- 配置：[`configs/study.yaml`](../../configs/study.yaml)；服务模板 [`configs/serve.yaml`](../../configs/serve.yaml)，评测模板 `configs/eval-*.yaml`。
- 模型（全参权重，直接只读加载，不复制）：
  - `start`：起点 8000 步合并模型 `unified-diarization-asr-20261001/tasks/merge-8000/attempts/001/artifacts/merged-model`
  - `realmix500`、`realmix1500`：`many-speaker-real-mix-acp-20261009/tasks/03-select/attempts/003/artifacts/models/checkpoint-{500,1500}`
  - `lrhalf500`、`lrhalf2000`：`many-speaker-lr-half-acp-20261009/tasks/03-select/attempts/003/artifacts/models/checkpoint-{500,2000}`
- 解码设置（服务端 `--override-generation-config`，ae 请求只带 `temperature=0`、`max_tokens=4096`）：
  - `default`：当前部署，不传覆盖（模型 `generation_config.json` 无 `repetition_penalty`，vLLM 取 1.0）
  - `rp1.05`：`repetition_penalty: 1.05`
  - `rp1.1`：`repetition_penalty: 1.1`（数据清洗阶段 Qwen 推理所用值）
- 评测集（全部完整集，`samples_per_source: 0`）：panel（AliMeeting dev 138 + NOTSOFAR dev 146）、many70（6–12 人合成 70）、meeting180（424）、chime6（CHiME-6 dev 150）；固定 338 条集只在每个模型的 default/r1 跑。

## 作业

19 个作业：5 个模型 × 3 种设置（r1），加 `start`、`realmix1500` 默认解码 r2、r3。每个作业起一个 vLLM 服务（独立 GPU、独立端口 8811–8814），评测集并发跑完后停服务。

## 解码设置核验

服务带 `--enable-log-requests`，vLLM 对每个请求记录实际生效的 `SamplingParams`。作业结束时从服务执行的 `logs/process.log` 解析全部请求，要求：请求数 ≥ 评测样本数、`repetition_penalty` 只有配置值一种、`temperature` 为 0；结果写入作业目录 `sampling-verification.json`，不通过即作业失败。

## 完成条件

全部作业 `result.json` 存在、解码核验通过、每个评测执行 W&B 远端核验完成。

## 进展与结果

| 执行 | 结果 | 说明 |
|---|---|---|
| 001 | interrupted（143） | 核验读错日志：`serve.log` 只有启动器的一行 JSON，vLLM 输出在服务执行的 `logs/process.log`；评测开始前中止，修复后重启。 |
| 002 | interrupted（143） | 完成 `start/default/r1`、`realmix1500/default/r1`、`start/rp1.05/r1`。`start/rp1.1/r1` 的 panel 评测失败：一条 NOTSOFAR 输出含 U+2028，AmphionEval 0.7.0 按 `str.splitlines()` 读 `worker-predictions.jsonl`，该条之后 52 条全部判失败（服务端 284 条都已返回）；同时旧驱动在作业失败后会停用该 GPU。中止后加入补分兜底（`scripts/rescore.py`，用 ae 自身 `score_sample`/`summarize`，对 4 个已完成评测复算与 ae 原结果逐字节一致）并改为只在服务起不来时停用 GPU。 |
| 003 | completed（0） | 沿用 002 的 3 个已完成作业（`reuse_jobs_dirs`），其余 16 个重跑全部完成；没有作业触发补分。19 个作业解码核验全部通过（default 只见 `repetition_penalty=1.0`，rp1.05/rp1.1 只见 1.05/1.1，`temperature=0.0`，`max_tokens=4096`，请求数 = 样本数）。W&B：https://wandb.ai/1016097967-amphion/open-audio-llm/runs/decode-loop-study-20261009-runs-003 （finished）；81 个评测 run 远端均为 finished。 |

执行完成；质量结论见 [03-analyze](../03-analyze/README.md) 与实验 [README](../../README.md)。

证据：各作业目录 `attempts/<编号>/artifacts/jobs/<模型>/<设置>/r<次>/` 下的 `result.json`、`sampling-verification.json`、`serve/attempts/*/logs/process.log`（含每个请求的 SamplingParams）与各评测执行。
