# 数据配方参数

[catalog_mixed.yaml](catalog_mixed.yaml) 按“数据来源 → 验证 → 组批 → 增强”阅读。
YAML 保留配方意图和关键约束，具体参数在这里查询；重复来源共用相同定义。

原生 Qwen3-ASR 热词训练使用 [qwen3_asr_hotwords.yaml](qwen3_asr_hotwords.yaml)，
启动与评估方法见 [原生模型配方](../../train/qwen3-asr/README.md)。

## 基础设置

| 参数 | 含义 |
|---|---|
| `seed` | 采样和增强的随机种子，结合 epoch 与重复位置保证可复现。 |
| `sampling_rate` | 统一音频采样率，单位 Hz，须匹配模型。 |
| `catalog`（可选） | Catalog 注册目录；相对路径从 YAML 所在目录解析。 |
| `roots`（可选） | 本机音频和清单的根目录映射文件。 |

启动脚本默认读取相邻 `audio-data-contract` 的 catalog 和 roots.json。
`AUDIO_DATA_CATALOG`、`AUDIO_DATA_ROOTS_FILE` 环境变量优先于 YAML。

## 数据来源

`train` 是训练来源列表，`validation` 是显式验证来源列表。验证关闭训练增强与随机热词。

| 来源参数 | 含义 |
|---|---|
| `dataset_id` | Catalog 中登记的数据集 ID。 |
| `version` | 数据集的固定版本。 |
| `split` | 该版本登记的划分，如 train、dev。 |
| `task` | 在线指令模板，如普通转写 asr、热词转写 asr_hotwords。 |
| `min_duration` / `max_duration` | 单条原始主音频的最短 / 最长时长，单位秒；须为慢速增强预留模型窗口。 |
| `samples` | 每轮选取数量上限；不足时取全部，省略时使用全部。 |
| `reps` | 选中数据每轮重复次数；默认交织权重为实际选中条数 × reps。 |
| `shard_rotation` | 按 epoch 轮换 samples 大小的逻辑分片，要求同时设置 samples；末片不足时从头补齐。 |
| `max_samples` | 固定取前 N 条合格样本，适合快速验证；不能与 samples 同时设置。 |
| `weight`（可选） | 正数相对交织权重，控制顺序概率，不增加每轮总量。 |

混合示例使用 samples/reps。改用显式 weight 时，给每个训练来源设置 weight，
并删除 samples/reps/shard_rotation；这两套混合配置不能同时使用。

### 固定配额回放

TS-ASR 配方见 [qwen3_asr_ts_replay.yaml](qwen3_asr_ts_replay.yaml) 与
[启动和验收说明](../../train/qwen3-asr/TS_ASR.md)。配置顶层 `replay` 后，
`weight` 改为每个窗口的样本配额权重；未启用 replay 的配方保持原来的有限交织语义。

```yaml
replay:
  epoch_samples: 20000
  window_samples: 200
```

两个字段均为正整数，epoch_samples 必须是 window_samples 的整数倍。每个来源都要设置
正数 weight，不能与 samples/reps/shard_rotation 混用。比例按最大余数法取整；
若窗口太小导致某来源配额为零则报错。来源内部随机无放回遍历，跨 epoch 延续位置，
遍历完才重新打乱回放。窗口控制全局样本数，不控制 token 或音频时长比例。
replay 模式使用 window_samples 作为分桶窗口，覆盖 batching.buffer_size。

## 组批

`batching` 仅用于 SFT。每个训练 batch 的音频槽位数相同。

| 参数 | 含义 |
|---|---|
| `max_duration` | 每 rank 的 batch 音频时长上限，单位秒；累计所有槽位并预算最慢变速。 |
| `max_samples` | 每 rank 每 batch 的条数上限，与时长限制同时生效。 |
| `num_buckets` | 时长桶数量。 |
| `buffer_size` | 每次读取多少个索引进行分桶；当前元数据仍驻留内存。 |
| `drop_last` | false 复制完整 batch 补齐各 rank 步数；true 丢弃不足一组 rank 的尾批，不表示丢弃所有小 batch。 |

## 热词采样

`hotwords` 控制候选词采样；包含热词指令的任务会在 prompt 中使用这些候选词。

| 参数 | 含义 |
|---|---|
| `max_hotwords` | 补入干扰词后的目标上限，已有真实热词不会被强制截断。 |
| `hotword_prompt_prob` | 提供候选热词的概率，范围 [0,1]。 |
| `hotword_miss_prob` | 每个真实热词从候选列表遗漏的概率，范围 [0,1]。 |

## 在线增强

`augmentation` 仅训练时启用；未配置的增强概率默认 0。

| 参数 | 含义 |
|---|---|
| `speed_prob` | 执行变速的概率，范围 [0,1]。 |
| `speed_factors` | 等概率选择的速度倍数；小于 1 更慢、更长。 |
| `spec_aug_prob` | mel 特征遮挡概率，仅 SFT 支持。 |
| `time_mask_width` | 时间遮挡宽度上限，单位特征帧。 |
| `frequency_mask_width` | 频率遮挡宽度上限，单位 mel 通道。 |
| `noise_prob` | 加噪概率；启用前必须填写 noise_sources。 |
| `noise_snr_db` | 均匀采样的信噪比范围，单位 dB；越低噪声越强。 |
| `rir_prob` | 混响概率；启用前必须填写 rir_sources。 |

顶层 `noise_sources` 和 `rir_sources` 分别提供噪声、RIR 来源，关闭对应增强时可为空。
每项使用前述 dataset_id、version、split 引用 Catalog 的 recordings 或 cuts，例如：

```yaml
noise_sources:
  - dataset_id: your-noise-dataset
    version: your-version
    split: train
```

替换为真实注册的数据集后再启用增强。完整训练流程与限制见
[在线训练数据](../../../docs/online_training_data.md)。
