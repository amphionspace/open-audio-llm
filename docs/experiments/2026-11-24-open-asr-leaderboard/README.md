# Open ASR Leaderboard 同口径评测：口径对齐与 11.17 交付包（2026-11-24）

承接[公开 ASR 榜单调研](../../research/2026-11-23-public-asr-leaderboards.md)的“建议路线”第 1 步：按 Hugging Face Open ASR Leaderboard（v1.16.2，2026-11-20）的数据、调用方式和打分方式测我们的模型，先用 Qwen3-ASR-1.7B 复现榜单数字，再测 11.17 交付包。

**结论**

- **口径已对齐。** 我们测的 Qwen3-ASR-1.7B 与榜单逐数据集只差 −0.11 到 +0.10 个点，英文 7 个公开集平均 5.73 对榜单 5.75，7 种语言分都在 ±0.04 以内。打分器与榜单仓库的原始代码在全部 26 个切片上逐位一致。所以以后提交前，可以先用这套评测预估榜单名次。
- **11.17 交付包现在不适合上榜。**
  - 英文 7 个公开集平均 6.12，比原版差 0.39，按同一口径排第 25/68，开源模型第 13。
  - 中文列（FLEURS + AISHELL-1）2.24，比原版 2.12 还差。AISHELL-1 靠训练集里的原句压到 0.38，干净的 FLEURS 中文反而从 2.70 退到 4.10。
  - 其他语言大幅退化：德、法、意、葡、荷的错误率是原版的 2–3 倍，多语种 7 语言平均 11.18 对 4.97。
- **找到一个能直接修的数据问题。** FLEURS 中文训练文本里有不读出来的英文括注，例如“孙德尔本斯（Sundarbans）”。最新终版文本 `fleurs_zh@legacy-20260804-final-v2-20261123` 中仍有 409/3246 条（12.6%）。11.17 学会了凭空输出这种括注：945 条测试里有 49 条，原版只有 11 条。只去掉括注，FLEURS 中文就从 4.10 降到 3.04，占它与原版差距的约 3/4。

## 结果

错误率 %，中文为 CER。“榜单”是榜单上 `Qwen/Qwen3-ASR-1.7B-hf` 的数字（v1.16.2，英文主表和各语言的数据集明细）。

| 数据集 | 榜单 Qwen3 | 我们测 Qwen3 | 差 | 我们测 11.17 | 11.17 − Qwen3 |
| --- | --- | --- | --- | --- | --- |
| AMI（清洗版） | 8.31 | 8.28 | −0.03 | 8.49 | +0.21 |
| Earnings22（分段，按场拼接） | 5.84 | 5.82 | −0.02 | 7.08 | +1.26 |
| GigaSpeech（清洗版） | 7.22 | 7.21 | −0.01 | 7.52 | +0.31 |
| SPGISpeech | 2.57 | 2.58 | +0.01 | 3.09 | +0.51 |
| URGENT24（加噪） | 9.93 | 9.82 | −0.11 | 10.77 | +0.95 |
| 印度口音对话（Monsoon） | 3.50 | 3.49 | −0.01 | 3.89 | +0.40 |
| VoxPopuli（清洗版） | 2.85 | 2.94 | +0.09 | 2.01 | −0.93 |
| **英文 7 个公开集平均** | **5.75** | **5.73** | −0.02 | **6.12** | +0.39 |
| FLEURS 中文 | 2.72 | 2.70 | −0.02 | 4.10 | +1.40 |
| AISHELL-1 | 1.51 | 1.51 | 0.00 | 0.38 ⚠️ | −1.13 |
| **中文列** | **2.12** | **2.10** | −0.02 | **2.24** | +0.14 |
| 德语（FLEURS / MCV） | 3.88 | 3.86 | −0.02 | 8.53 | +4.67 |
| 法语（FLEURS / MCV / MLS） | 5.63 | 5.64 | +0.01 | 11.85 | +6.21 |
| 意大利语 | 5.42 | 5.38 | −0.04 | 14.90 | +9.52 |
| 西班牙语 | 3.70 | 3.69 | −0.01 | 4.50 | +0.81 |
| 葡萄牙语（FLEURS / MLS） | 6.16 | 6.17 | +0.01 | 12.01 | +5.84 |
| 荷兰语 | 7.89 | 7.93 | +0.04 | 24.25 | +16.32 |

逐数据集明细见 [compare.md](compare.md)，打分器输出见 `reports/`。

**放到榜单里看**（英文按 7 个公开集平均比较，私有集两列我们测不到）：

| 位置 | 英文公开集平均 | 中文列 |
| --- | --- | --- |
| 全榜第 1 | 4.19（zoom/scribe_v2.1_pro，API） | 1.89（Azure，API） |
| 开源第 1 | 5.15（Musci-ASR-qwen-1019，Qwen 系微调） | 2.12（Qwen3-ASR-1.7B） |
| Qwen3-ASR-1.7B 原版 | 5.75（第 17） | 2.12 |
| **AmphionASR 11.17** | **6.12（第 25，开源第 13）** | **2.24** |

⚠️ AISHELL-1：测试集 96.5% 的句子原文出现在训练集，主要来自 AISHELL-2 训练集，见[调研记录 3.1](../../research/2026-11-23-public-asr-leaderboards.md)。榜单版 FLEURS 中文与训练集零重合。

## 问题与下一步

按影响排序：

1. **FLEURS 中文训练文本带不读出来的括注**：去掉 `fleurs_zh` 训练文本里的这类括注（并检查其他 FLEURS 语种和同类来源），交给数据侧在下一版文本里处理。
2. **AISHELL-1/3、MagicData 文本重合**：发布版训练去掉与公开测试集原句相同的条目；否则中文列里 AISHELL-1 这一半会被看成刷榜。
3. **英文不如原版**：7 个公开集有 6 个变差，Earnings22（+1.26）、URGENT24 加噪（+0.95）、SPGISpeech（+0.51）最多。要上英文榜，至少要先回到原版水平。
4. **多语种遗忘**：德、法、意、葡、荷退化 2–3 倍，西语也差 0.8。要打多语种榜，就得在训练里保住这些语言，例如回放多语种数据或与原版合并。按 11-19 的决定，交付侧仍以中英为先。
5. **语种识别退化**：冒烟测试中，不强制语言时 11.17 把德语语音标成 `language English`（文字本身是德语）。榜单强制语言，不影响上面的分数，但会影响自动识别语种的部署场景。

## 做了什么

| 项目 | 内容 | 位置 |
| --- | --- | --- |
| 数据 | 榜单公开测试集原样镜像：英文 7 个默认集 + URGENT24 干净版，多语种 7 种语言（中、德、法、意、西、葡、荷）的 FLEURS、Common Voice、MLS、AISHELL-1；固定 HF revision，下载后逐文件校验 sha256；26 个切分条数与 HF 一致，共 140,295 条、约 287 小时 | 数据集 `open_asr_leaderboard@v1-20261124`；audio-data-contract 分支 `feat/open-asr-leaderboard-mirror`（`scripts/open_asr_leaderboard/build.py`） |
| 调用 | 榜单用 qwen_asr 跑 Qwen3 系模型，每条强制语言（提示词以 `language X<asr_text>` 结尾）；原生评测器新增 `force_language`，服务换一个评测用模板（不传参数时与原模板逐字节相同）。英文 `max_new_tokens` 512、多语种 256，与榜单脚本一致 | amphion-eval 分支 `feat/open-asr-leaderboard` |
| 打分 | 榜单仓库 commit `c4696193` 的规范化器原样搬入；英文合并拆写复合词，中文按字并用 wetext 读数字，Earnings22 按场拼接 | `ae open-asr-leaderboard score` |
| 推理 | 部署栈 vLLM 0.18.0：Qwen3-ASR-1.7B 原版；11.17 交付包用包内 `serve.sh asr`（MTP k=2）；各 3 卡数据并行 | `/workspace/data/ae-runs/open-asr-leaderboard-20261124/` |

不覆盖：英文私有集（榜单英文平均里有两列私有）、印地语（私有集，lattice 打分）。榜单这一条目是 transformers 版 `Qwen3-ASR-1.7B-hf`，我们用 vLLM，两者有后端差异。

## 过程记录

- 第一次启动失败：after-training 会去掉评测子进程的 `PYTHONPATH`，子进程用的是环境里旧版 amphion-eval，不认识 `--force_language`。改成把分支 editable 装进独立 venv 后重跑；失败的两个 run 保留在 `runs/failed-20261124/`。
- 单卡服务时 GPU 利用率只有 50% 左右、vLLM 队列一直排着 50 多条请求，瓶颈在单个服务的并发和前端音频预处理。停掉后改为每个模型 3 卡数据并行、3 个 API 前端，原 run 用 `--resume` 续跑。
- 续跑没有沿用中断前的约 4.9 万条预测：`--resume` 只读已合并的 `predictions.jsonl`，并清空 `worker-predictions.jsonl`；被强停的进程还没合并，这部分推理重做了一遍。

## 证据

| run | 模型 | 部分 | 条数 | W&B |
| --- | --- | --- | --- | --- |
| `run-20261124T075202Z-3248c0fb` | Qwen3-ASR-1.7B | 英文 | 70,985，全部成功 | [链接](https://wandb.ai/1016097967-amphion/open-audio-llm/runs/run-20261124T075202Z-3248c0fb) |
| `run-20261124T075202Z-998e1fe3` | Qwen3-ASR-1.7B | 多语种 | 69,310，全部成功 | [链接](https://wandb.ai/1016097967-amphion/open-audio-llm/runs/run-20261124T075202Z-998e1fe3) |
| `run-20261124T075202Z-c6c13e39` | 11.17 交付包 | 英文 | 70,985，全部成功 | [链接](https://wandb.ai/1016097967-amphion/open-audio-llm/runs/run-20261124T075202Z-c6c13e39) |
| `run-20261124T075202Z-7758c3b6` | 11.17 交付包 | 多语种 | 69,310，全部成功 | [链接](https://wandb.ai/1016097967-amphion/open-audio-llm/runs/run-20261124T075202Z-7758c3b6) |

- run 目录：`/workspace/data/ae-runs/open-asr-leaderboard-20261124/runs/<模型>/runs/<run>`；配置、服务脚本、日志在同一活动目录。
- 4 个 run 都在 W&B 远端核对过：状态 finished，原生指标和 `open_asr_leaderboard/*` 榜单口径分数都在 summary 里。
- 逐位对照：用榜单仓库原始的 `normalizer/eval_utils.score_manifest` 重算同一批预测，与 `ae open-asr-leaderboard score` 的插入、删除、替换计数在全部切片上相同（[`xcheck.py`](xcheck.py)：加载榜单仓库 commit `c4696193` 的代码，把它只用于下载数据的 `datasets`、`huggingface_hub` 换成空模块）。对照表由 [`compare.py`](compare.py) 生成。
- FLEURS 中文括注：11.17 预测中带拉丁文括注的有 49/945 条，原版 11/945；训练文本 `fleurs_zh@legacy-20260804-final-v2-20261123` 有 409/3246 条。
- 训练集重合（用 `ts-robust-062` 的训练配置）：榜单版 FLEURS 中文 0/945，AISHELL-1 6925/7176。
