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

### 更换训练配方前的数据检查

旧训练继续运行时，先固定新配方、Catalog、roots 和源码快照，并执行下面的数据检查。
该命令不加载模型，不启动试训；它核对可复用缓存，准备缺失索引，实例化真实
`CatalogBatchSampler` 检查配额，再读取每个训练/验证来源的最短和最长样本。

```bash
CUDA_VISIBLE_DEVICES='' python -m open_audio_llm.data.catalog_cache \
  --data_config /path/to/new-run/train-data.yaml \
  --reuse-config /path/to/old-run/train-data.yaml \
  --workers 4 --world-size 2 --batch-size 8 \
  --preflight-report /path/to/new-run/data-preflight.json
```

`AUDIO_DATA_METADATA_CACHE` 指向已有缓存目录；`--batch-size` 与正式启动参数一致。
旧配方使用其文件中固定的 Catalog/roots，不受新任务导出的同名环境变量覆盖。
只有索引输入、版本、过滤和处理参数一致的已完成缓存才会复用，不清空旧目录。
新增无关路径别名或调整字典键顺序不再重建旧语料；有序 shard 列表的顺序仍属于输入。

必须等命令成功，检查报告中的每桶 `quota`、`quotas_by_dataset` 和读取结果，再等旧任务
保存完整 checkpoint、停止旧任务、以其权重启动新配方。数据检查失败时旧训练继续运行。
正式启动入口也会在 DDP 前执行同一检查；可设置 `PREVIOUS_DATA_CONFIG` 指向旧配方。
配比按样本条数计算。窗口太小时检查会失败，不会静默删除小分桶或修改权重。
本次事件配方使用 10,000 条窗口：新事件 3,500、旧合成 500、真实会议 4,000、普通 ASR 2,000。

### 冻结源码和 W&B 同步

`source-provenance.json` 可以只包含 `source_origin` 和 `files`（相对路径到 SHA-256 的映射）；
同步器也读取已有的 `files_sha256`、`run_files_sha256`。没有精确 Git 版本时不要求
`git_commit`。`git_commit` 仅用于确实对应该提交的源码；冻结目录包含提交之外的改动时，
将原始基线记录为 `base_git_commit`，以冻结文件哈希标识实际版本。不得用同步工作区的
HEAD 替代来源。修改同步入口时保留旧入口及其哈希，训练源码快照保持不变。

同步失败后只恢复同步进程，继续使用同一个 W&B run ID。同步器先读取远端 `sync_event`
历史，再补录未确认的本地事件；单写入进程锁防止两个同步器同时上传。
交付前检查远端实际 step/loss 至少两次，确认持续增长，并检查历史事件没有重复。

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

每次启动实验都必须同时拉起 W&B 持续同步。默认使用 entity
`1016097967-amphion`、project `open-audio-llm`。训练生成 `args.json` 和启动检查记录后，
立即执行下列同步命令；核验 `wandb-sync/run.json` 的链接及远端实际指标后再报告接入成功。
已有实验漏启同步时，补传历史并继续跟随，不重启训练。

`open_audio_llm.scripts.sync_wandb` 可独立读取运行目录中的训练、性能日志和已完成的
固定评测。无需修改或重启训练；安装 `tracking` 可选依赖，或在独立 Python 环境安装
`wandb==0.30.0` 与 `PyYAML>=6`。先通过 `wandb login` 配置凭据，不把 key 放入脚本。

```bash
PYTHONPATH=src python -m open_audio_llm.scripts.sync_wandb \
  --run-dir runs/qwen3-asr-sot-v2-2gpu-20260916 \
  --entity 1016097967-amphion --project open-audio-llm --follow
```

不加 `--follow` 时只补录历史。run ID 默认取运行目录名；重启同步时从远端历史识别
已上传事件。一个运行目录只允许一个同步进程。训练和评测使用 `global_step` 作横轴，
评测晚于训练日志到达也能补传。识别错误率、人数/归属准确率、覆盖率均以百分数显示；
中文保持差值以百分点显示。`perf/rank0` 是 rank 0 的性能统计，不当作整机总吞吐。

四卡阶段和两卡恢复阶段分别建 run，后者记录来源检查点、恢复步数和优化器是否重建。
仅上传选定的配置、汇总指标与数据/代码版本指纹；原始音频、转写正文和模型权重继续
留在本地。同步进程的运行状态表示日志同步状态，不能单凭 W&B 的 Running 标记判断
训练是否仍在更新，应同时看 `latest_training_step`。

## 每次发言带时间戳

源音频的强制对齐在 AmphionData 完成。本项目可直接消费其 `align` 输出，复用已经
合成的混音，不需要重新合成音频。当前目标为每次发言一行，例如：

```text
[S1][0.32-2.48] 你好。
[S2][1.04-3.20] Good morning.
[S1][3.60-4.80] 再见。
```

每次发言取原句第一个字词的起点和最后一个字词的终点，保留完整原文；当前没有按
句内静音进一步切句，也不输出逐字时间戳。源对齐的时间坐标必须是
`source_segment_start`，混音边界等于 `segment.start + item.start/end`，不能再减
源录音裁剪起点。按实际对齐后的首次发声重新编号，原 `metadata.segments` 不改写。

先冻结训练侧查询索引，再创建带时间戳的训练视图。`--input` 可重复，合并 train、
dev、test 的来源结果；不同目录不能包含重复的来源版本和样本 ID。

```bash
python -m open_audio_llm.data.sot_alignment \
  --input /path/to/source-alignment-train \
  --input /path/to/source-alignment-dev \
  --input /path/to/source-alignment-test \
  --output /path/to/timed-sot/source-alignments.sqlite
python examples/train/qwen3-asr/prepare_sot.py \
  --data-root "$SOT_DATA_ROOT" \
  --catalog "$AUDIO_DATA_CONTRACT_ROOT/catalog" \
  --roots "$AUDIO_DATA_CONTRACT_ROOT/roots.json" \
  --alignment-index /path/to/timed-sot/source-alignments.sqlite \
  --output /path/to/timed-sot
```

索引旁的 JSON 保存来源计划、模型配置、分片校验和与状态计数；训练配置固定索引
SHA-256，数据缓存也包含索引身份。索引允许收录待复核状态，但只有 `aligned`
结果能用于目标构建。训练数据加载会检查来源版本、音频路径、切片范围、划分、原文
及实际解码时长；缺失、待复核或不一致会明确失败，不静默丢掉发言或样本，也不使用
整段音频边界兜底。视图配置生成成功不代表全量对齐已通过，启动前须确认完整覆盖。
已有混音若使用原版本，不能把另一 clean 版本的对齐直接贴上去。

时间戳样本禁用变速和 RIR 增强，普通 ASR 回放及 60/30/10 采样比例、KL 权重保持
不变。两卡试训可以沿用现有入口；从上一轮最终模型初始化新优化器和学习率计划：

```bash
export MODEL=/path/to/previous-run/training/checkpoint-60000
export DATA_CONFIG=/path/to/timed-sot/train-data.yaml
export OUTPUT_DIR=/path/to/timed-sot/training
export CUDA_VISIBLE_DEVICES=0,1 NPROC_PER_NODE=2 MAX_STEPS=2000
bash examples/train/qwen3-asr/train_sot.sh \
  --gradient_accumulation_steps 4 --warmup_steps 100
```

此例是 2,000 步试训；encoder/aligner/LLM 保持原学习率 `2e-5/2e-5/1e-5`，
全参数更新，只保存模型。正式启动还需按实验配置接上普通 ASR teacher、固定开发集
基线与定期评测。中文、英文、中英混合及 dev/test 缺失时，不应直接缩成英文训练。

评测会去除时间字段、按人合并原文后计算原有 cpCER/cpWER/cpMER、人数与归属指标。
另报时间格式正确率、匹配发言的起止边界平均绝对误差（秒）、匹配覆盖率，以及起止
误差均不超过 0.5 秒的 precision/recall/F1。先按转写编辑距离匹配说话人，再匹配
发言；时间误差不参与匹配。MAE 必须结合覆盖率查看，漏掉的发言会降低 recall。
这些是自动对齐参考下的发言边界指标，不是逐字时间精度或 DER；无时间戳的旧输出
仍可比较转写分数，但时间格式与召回记为失败。W&B 同步保留秒和百分数的单位区别。
