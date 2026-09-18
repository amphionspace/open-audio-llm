# 全说话人转写

输入一段混音，按首次发声顺序输出 `[S1]`、`[S2]` 等标签，每人一行完整文字。
任务为 `speaker_attributed_asr`，没有 enrollment，也不拼接静音前缀；单人也输出 `[S1]`。

## 数据准备与覆盖

数据加工和合成在相邻的 `../AmphionData` 仓库维护，公共命令为
`amphion-data multispeaker prepare|generate`。本项目通过
`synthesize_sot.sh` 调用该命令；`prepare_sot.py` 只准备训练采样配置及回放划分。
`audio-data-contract` 负责数据契约与登记，训练和评测通过 Catalog 消费成品。
v2 配方覆盖 1～5 人、中文/英文/中英同场，以及 0、0～20%、20～40%、40～65%、
65～100% 的活动重叠率；每种人数等量，多人部分 20% 为中英同场。最高重叠组还要求
实际存在所有人同时活动的帧。源语句完整保留，优先使用适用 clean 训练版本。

配方借鉴 [MOSS-Transcribe-Diarize 0.9B §3.2](https://arxiv.org/pdf/2601.01554v7)
的轮换发言与时间轴调度；保留本项目 28 秒窗口、完整原句和不额外加噪的约束。
具体重叠参数和源语料策略见
[`AmphionData/docs/multispeaker-synthesis.md`](../../../../AmphionData/docs/multispeaker-synthesis.md)。

仓库按相邻目录引用，可用 `AMPHION_DATA_ROOT` 覆盖位置。合成使用 AmphionData
自己的 Python 3.11 环境，`AMPHION_DATA_PYTHON` 可覆盖解释器路径；训练环境不需要
安装 AmphionData。已有完整产物时可直接跳到下方训练视图准备。

```bash
export AMPHION_DATA_ROOT=/path/to/AmphionData
export AUDIO_DATA_CONTRACT_ROOT=/path/to/audio-data-contract
export SOT_DATA_ROOT=/path/to/sot-multispeaker/synthetic-v2-20260915

# 在 AmphionData 的独立环境安装 CPU 合成依赖。
(cd "$AMPHION_DATA_ROOT" && uv sync --extra multispeaker)
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
bash examples/train/qwen3-asr/synthesize_sot.sh prepare \
  --recipe "$AMPHION_DATA_ROOT/src/amphiondata/multispeaker/recipe-v2.json" \
  --catalog "$AUDIO_DATA_CONTRACT_ROOT/catalog" \
  --roots "$AUDIO_DATA_CONTRACT_ROOT/roots.json" \
  --output "$SOT_DATA_ROOT" --workers 4
bash examples/train/qwen3-asr/synthesize_sot.sh generate \
  --output "$SOT_DATA_ROOT" --workers 32 \
  --train 2000000 --dev 10000 --test 10000 --shard-size 500
```

合成命令保留 v1 默认参数；上面的 v2 配方和样本数需显式指定。迁移不改变 v1/v2
版本号、说话人划分或采样算法，既有成品可以继续使用。输出的 `catalog.jsonl`
是本地可消费声明，注册到 `audio-data-contract/catalog/` 仍走独立发布流程。

```bash
export AUDIO_DATA_CONTRACT_ROOT=/path/to/audio-data-contract
export PYTHONPATH="$PWD/src:$AUDIO_DATA_CONTRACT_ROOT/src"
python examples/train/qwen3-asr/prepare_sot.py \
  --data-root /path/to/sot-multispeaker/synthetic-v2-20260915 \
  --catalog "$AUDIO_DATA_CONTRACT_ROOT/catalog" \
  --roots "$AUDIO_DATA_CONTRACT_ROOT/roots.json" \
  --output /path/to/sot-v2-training-view
```

只接受已经生成完成的数据。准备脚本从实际成品发现条件，兼容 v1 的 12 类与 v2 的
62 类；训练权重按成品条数计算，不将单人或中英条件漏掉。普通 ASR 回放排除合成
dev/test 说话人，v2 复用 v1 的源池划分，避免旧训练污染新开发/测试集。

抽样比例为 60% 全说话人转写、30% 中文普通 ASR、10% 英文普通 ASR。
每 2,500 条为一个固定配额窗口，完整表示 v2 各条件和普通 ASR 来源的权重。
基座 KL 权重为 2，仅约束普通 ASR；这些措施需要通过固定普通 ASR 评测验证，
不能保证中文能力完全不退化。

## 中英输入与评分

中英同场为不同人分别说中文、英文，所有正文保留原语言，不覆盖同一人的句内切换。
成品语言为 `zh-en`，`metadata.primary_language` 按源语句总时长选择 `zh` 或 `en`，
用于原生 Qwen3-ASR 的单语言输出头。混合任务提示明确要求保留两种原文、不翻译。

`open_audio_llm.eval.ts_asr` 同时支持普通 ASR 和此任务。纯中文按 cpCER、纯英文按
cpWER，中英混合按 cpMER 评分：中文字与英文词各计一个单位，再匹配说话人使总编辑
距离最小。另报告人数准确率、格式正确率和空输出率。不同人数、语言、重叠区间分别
报告；v2 的总体分数不能直接与条件分布不同的 v1 总分比较。

`speaker_attribution` 单独衡量可判定字词的说话人归属，方法版本为
`unique_reference_units_v1`。只统计在参考文本中属于唯一说话人的字词；若预测总次数
超过参考次数，排除该字词，避免将重复插入当作正确识别。先最小化 cpWER 的编辑距离，
若多个说话人排列同分，再取归属正确数最大的排列；无标签正文不获得归属正确的计数。

- `accuracy`：归属正确的字词数 / 可判定且在预测中出现的字词数，越高越好。
- `coverage`：上述分母 / 全部参考字词数。共享字词、重复过量、漏识别都会降低覆盖率。
- `correct_units`、`scored_units` 为分子、分母；`unique_reference_units` 是参考中的
  唯一归属字词数，`excess_reference_units` 是其中因预测重复过量而排除的参考字词数。

汇总按字词数加权，没有可判定字词时准确率为 `null`。当前输出按人汇总、没有词级
时间戳，这个统计不验证字词的发生时间或顺序，也不评价多人共有字词的归属，因此是
条件准确率，不能当作完整 WDER/DER、声纹身份分类或首次发声顺序准确率。必须与
覆盖率、cpWER/cpCER/cpMER 一起查看，不能用它替代转写完整性评价。

正常评测自动包含此统计；旧预测可以在 CPU 上回算，无需加载模型，输出独立报告：

```bash
python -m open_audio_llm.eval.sot \
  --predictions /path/to/eval/sot/predictions.jsonl \
  --output /path/to/eval/sot/speaker-attribution.json
```

新增开发集有 62 类，建议训练中固定抽每类 8 条（496 条），另保留固定的普通 ASR
基线；正式报告再用完整测试集。更换数据或代码快照时先重新生成相同协议的基线。

训练入口为 `train_sot.sh`，四卡全参数，encoder/aligner/LLM 学习率分别为
`2e-5/2e-5/1e-5`，仅保存模型。设置 `MODEL`、`DATA_CONFIG`、`OUTPUT_DIR` 后启动，
首次训练应带 `--warmup_steps 300`。已有运行使用其冻结的代码与配置；准备新数据不会
使正在运行的训练自动切换，切换需要在保存点启动新的运行并明确记录模型来源。

## W&B 实验记录

`open_audio_llm.scripts.sync_wandb` 可独立读取运行目录中的训练、性能日志和已完成的
固定评测。无需修改或重启训练；安装 `tracking` 可选依赖，或在独立 Python 环境安装
`wandb==0.30.0` 与 `PyYAML>=6`。先通过 `wandb login` 配置凭据，不把 key 放入脚本。

```bash
PYTHONPATH=src python -m open_audio_llm.scripts.sync_wandb \
  --run-dir runs/qwen3-asr-sot-v2-2gpu-20260916 \
  --entity YOUR_ENTITY --project open-audio-llm --follow
```

不加 `--follow` 时只补录历史。run ID 默认取运行目录名；重启同步时从远端历史识别
已上传事件。一个运行目录只允许一个同步进程。训练和评测使用 `global_step` 作横轴，
评测晚于训练日志到达也能补传。识别错误率、人数/归属准确率、覆盖率均以百分数显示；
中文保持差值以百分点显示。`perf/rank0` 是 rank 0 的性能统计，不当作整机总吞吐。

四卡阶段和两卡恢复阶段分别建 run，后者记录来源检查点、恢复步数和优化器是否重建。
仅上传选定的配置、汇总指标与数据/代码版本指纹；原始音频、转写正文和模型权重继续
留在本地。同步进程的运行状态表示日志同步状态，不能单凭 W&B 的 Running 标记判断
训练是否仍在更新，应同时看 `latest_training_step`。
