# Qwen3-ASR-1.7B 热词微调

TS-ASR 与中文通用识别能力保持使用独立的 [TS-ASR 回放配方](TS_ASR.md)。

使用原生 Qwen3-ASR 权重和 Catalog 数据，在取样时解码音频、采样候选热词并增强，不生成 ShareGPT 文件。`amphion_asr_1.7b` 是现有插件的注册名，实际模型仍是 Qwen3-ASR。

环境需安装项目的 `data`、`swift` 依赖，以及 `qwen3-asr` 可选依赖组：`qwen-asr` 提供原生模型，`rapidfuzz` 计算评估编辑距离。此次运行环境为 Python 3.11、torch 2.10.0+cu128、transformers 4.57.6、ms-swift 4.1.0、qwen-asr 0.0.6、PEFT 0.18.1；其他版本未验证。

`qwen3-asr` 依赖组固定 `qwen-asr==0.0.6`：原生音频卷积 hook、监督位置投影和并行编码依赖该版本内部接口，升级时需要重新验证这些路径。

```bash
export MODEL=/path/to/Qwen3-ASR-1.7B
export PYTHON=/path/to/environment/bin/python
export CUDA_VISIBLE_DEVICES=2,3
export AUDIO_DATA_CONTRACT_ROOT=/path/to/audio-data-contract
export OUTPUT_DIR=$PWD/runs/qwen3-asr-hotwords
bash examples/train/qwen3-asr/train.sh
```

默认读取 [qwen3_asr_hotwords.yaml](../../configs/data/qwen3_asr_hotwords.yaml)。数据字段含义见 [数据配置说明](../../configs/data/README.md)。此配方取每个训练来源的前 10000 条合格样本，再动态混合、分桶；不是全量语料的随机抽样。`evaluation` 与 `validation` 使用同样的来源字段，但仅评估命令读取 `evaluation`。

热词放在 system 中：`Hotwords: Alice,Bob`；user 只有 `<audio>`，assistant 为 `language English<asr_text>Hello Alice.`。中文标签为 `Chinese`。普通 ASR 或采样时丢弃热词时，system 为空。原生接入支持 `asr` / `asr_hotwords`，以及将双槽位拼接成单音频的 `ts_asr`，自动根据训练模型类型选择格式。

| 训练参数 | 默认值与含义 |
| --- | --- |
| `MODEL` / `PYTHON` | 必填模型路径；Python 默认使用当前环境 |
| `DATA_CONFIG` | 可覆盖默认 YAML 路径 |
| `AUDIO_DATA_CONTRACT_ROOT` | 数据契约项目路径；默认相邻目录 |
| `AUDIO_DATA_CATALOG` / `AUDIO_DATA_ROOTS_FILE` | 可分别覆盖 Catalog 目录与本机根路径映射 |
| `CUDA_VISIBLE_DEVICES` / `NPROC_PER_NODE` | 默认 GPU 2、3 / 两个训练进程；修改时需保持一致 |
| `MASTER_PORT` | 默认 29523，分布式进程通信端口 |
| `OUTPUT_DIR` / `MAX_STEPS` | 输出目录 / 默认 500 次优化器更新 |
| `OMP_NUM_THREADS` / `OPENBLAS_NUM_THREADS` | 默认 2 / 1，控制 CPU 线程数 |
| LoRA rank / alpha | 32 / 64，适配器秩与缩放；仅训练语言模型线性层 |
| 冻结范围 | 冻结音频编码器、连接层，保留预训练音频特征 |
| batch / 梯度累积 | 每卡最多 8 条、累积 1 次；双卡通常每步 16 条，受 YAML 时长预算限制 |
| learning rate / scheduler | `5e-5`，前 5% 步数预热，随后 cosine 衰减 |
| dtype / attention | bfloat16 / SDPA |
| max length | 1024 个 token，包含音频占位 token 与文字 |
| gradient checkpointing | 默认关闭；双 A800 80GB 的对照实验表明重计算降低吞吐。冻结音频塔的 checkpointing 单独关闭 |
| DDP unused 参数扫描 | 关闭；此配方所有可训练参数都参与前向 |
| `PERFORMANCE_LOGGING` | 默认 true，记录实际吞吐、分阶段耗时及每秒 GPU 利用率 |
| data workers | 每卡 2 个进程，执行在线解码和增强 |
| save / eval / logging | 每 250 步保存并计算 dev loss，保留 2 个 checkpoint；每 10 步记录日志 |
| seed | 42，控制训练和数据采样随机性 |

性能字段解释和对照结果见 [训练性能日志](../../../docs/training_performance.md)。

脚本末尾的额外参数传给 ms-swift，例如 `--learning_rate 2e-5`。断点继续可传 `--resume_from_checkpoint /path/to/checkpoint-250`。

## 固定条件评估

```bash
export PYTHONPATH="$PWD/src:$AUDIO_DATA_CONTRACT_ROOT/src"
export AUDIO_DATA_CATALOG="$AUDIO_DATA_CONTRACT_ROOT/catalog"
export AUDIO_DATA_ROOTS_FILE="$AUDIO_DATA_CONTRACT_ROOT/roots.json"
CUDA_VISIBLE_DEVICES=2 "$PYTHON" -m open_audio_llm.eval.qwen3_asr \
  --model "$MODEL" \
  --data_config examples/configs/data/qwen3_asr_hotwords.yaml \
  --output_dir "$OUTPUT_DIR/eval-base"

CUDA_VISIBLE_DEVICES=3 "$PYTHON" -m open_audio_llm.eval.qwen3_asr \
  --model "$MODEL" --adapter /path/to/checkpoint-500 \
  --data_config examples/configs/data/qwen3_asr_hotwords.yaml \
  --output_dir "$OUTPUT_DIR/eval-tuned"
```

`--adapter` 可选，省略时评估基础模型；推理时在内存合并 LoRA，不修改原始权重。每个来源默认 `--samples_per_source 256`，一半包含有效热词标注，另一半不含；`--seed 42` 固定抽样，`--distractors 10` 为每条添加 10 个不在参考答案中的干扰词。`--batch_size 8` 控制推理批量。

每条音频分别运行无提示、热词提示两种条件。热词提示包含标注正例，因此是已知候选词条件下的能力测试，不能代表真实检索效果。评估子集经过分层抽样，也不能当作全量测试集成绩。候选词不会作为额外输出标签。

输出 `predictions.jsonl` 保留样本 ID、参考答案、预测和候选词，`summary.json` 给出英文 WER / 中文 CER、热词召回率和干扰词误报率。文本统一 NFKC、小写、去标点；英文按词、中文按字计算错误率。热词按英文词边界或中文子串匹配。基础模型与微调模型使用相同的抽样、候选词及解码条件，不强制语言，最多生成 256 个 token。
