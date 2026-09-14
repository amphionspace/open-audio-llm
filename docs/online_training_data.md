# 在线训练数据

训练直接消费 `audio-data-contract` 的 Catalog。数据配置使用 YAML，优先使用
适用的 clean 版本，不需要生成 ShareGPT、增强音频或预编码特征。

## 后训练清洗要求

有适用于同一训练任务和划分的 clean 版本时，配方显式选择该固定版本并设置
`require_clean_pass: true`。Lhotse 按 `custom.clean.pass` 为布尔 `true` 且时间范围
有效过滤；AudioRecord 按 `metadata.clean.pass` 为布尔 `true` 过滤。失败、缺少标记
或伪布尔值不通过；整个所选 clean 来源没有合格记录则报错，不偷偷混入失败样本。

没有适用 clean 版本时允许使用原训练集，省略 `require_clean_pass` 或设为 `false`。
不会因缺少标记、包含多路音频或 Catalog 标记未重新审核而阻断这些来源。
只有 clean 测试集不代表存在 clean 训练版本，不能把测试集拿来训练。当前配方固定
引用版本，不在运行时按名字猜测或自动切换版本，避免改变数据和采样恢复语义。

清洗通过和数据格式/文件完整性校验是不同事实，不把未审核数据描述为已清洗。
显式 clean 过滤不会复用旧的未过滤 AudioRecord 缓存。此策略同样适用于 SFT 和 GRPO。
固定开发/测试集保留原样用于能力对照，缓存预构建区分训练和验证模式。

```text
Catalog + roots → Lhotse cut / AudioRecord 元数据
→ 按来源限量、重复、epoch 分片轮换 → 加权交织
→ 按时长和音频槽位数组 batch → DDP 各 rank 分配完整 batch
→ 按需加载波形、增强、渲染 messages
→ 内存 WAV → 模板提特征 / SpecAugment → collator → model
```

## 配置和启动

```bash
pip install -e /222042021/mingdong/workspace/audio-data-contract
pip install -e '.[swift,data,hf]'

CUDA_VISIBLE_DEVICES=2,3 NPROC_PER_NODE=2 \
MODEL=/path/to/open-audio-llm-hf-checkpoint \
DATA_CONFIG=examples/configs/data/catalog_mixed.yaml \
MAX_STEPS=1000 SAVE_STEPS=200 EVAL_STEPS=200 \
DATALOADER_NUM_WORKERS=2 \
OUTPUT_DIR=runs/catalog-mixed \
bash examples/train/sft/train.sh
```

脚本默认读取相邻的 `audio-data-contract/catalog` 和 `roots.json`。
`AUDIO_DATA_CONTRACT_ROOT` 可覆盖仓库位置；`AUDIO_DATA_CATALOG` 和
`AUDIO_DATA_ROOTS_FILE` 可分别覆盖这两个路径。也可在 YAML 写 `catalog` / `roots`，
直接使用 Python 训练入口时，相对路径按 YAML 所在目录解析，环境变量优先。
Catalog 和数据注册继续由契约仓库维护。

`examples/configs/data/catalog_smoke.yaml` 是少量样本的快速验证配置；
`examples/configs/data/catalog_mixed.yaml` 是完整的 SFT 混合示例：

```yaml
seed: 42
sampling_rate: 16000
train:
  - dataset_id: librispeech
    version: icefall-20260908
    split: train
    min_duration: 0.5
    max_duration: 25
    samples: 10000
    reps: 1
    shard_rotation: true
  - dataset_id: aishell
    version: icefall-20260908
    split: train
    min_duration: 0.5
    max_duration: 25
    samples: 10000
    reps: 2
    shard_rotation: true
validation:
  - dataset_id: librispeech
    version: icefall-20260908
    split: dev
    max_duration: 25
    max_samples: 64
batching:
  max_duration: 60
  max_samples: 8
  num_buckets: 30
  buffer_size: 10000
  drop_last: false
augmentation:
  speed_prob: 0.9
  speed_factors: [0.9, 1.0, 1.1]
  spec_aug_prob: 0.5
  time_mask_width: 30
  frequency_mask_width: 16
```

数据源支持 `cuts_artifact(s)`、`recordings_artifact(s)` +
`supervisions_artifact(s)`、`manifest_dir_artifact`，以及
`records_artifact`（`audio-record/1.0`）。录音按 supervision 切段，
音频槽位按照契约渲染顺序传入。来源可配置 `task`；在线热词参数放在顶层
`hotwords`，例如 `max_hotwords`、`hotword_prompt_prob`。

## 与 icefall 一致的混合语义

- `samples`：过滤后每轮选取的数量上限；未配置则使用整个来源。
- `reps`：选中部分在每轮重复的次数，默认 1。数量不足 samples 时按实际条数计算。
- 默认 mux 权重为 `实际选中数量 × reps`。以上例子每轮基础配额为 10000:20000。
- `shard_rotation: true`：按 samples 大小切逻辑分片，每个 epoch 轮换一片。
  最后一片不足时从头补齐，使每轮配额稳定；多轮覆盖整个来源。
  不需要离线切分清单，但当前仍会将该来源的全部元数据载入内存。
- 不启用轮换时 samples 固定取前 N 条。旧 `max_samples` 仍可用于快速截取，
  不能与 samples 同时配置。
- 或者给**每个**来源指定正数 `weight`；不能与 samples/reps 混用。
  与 icefall 的有限 `CutSet.mux` 一样，weight 控制来源交织的概率，各来源耗尽后退出，
  最终遍历全部选中数据。提高 weight 不等于增加该来源总样本数；增加总量使用 reps。

分桶会在 buffer 内调整交织顺序，不保证每个 batch 的来源比例完全相同。
未实现随 step 自动变化的权重调度。

## 动态 batch

`batching.max_duration` 是每个 rank 的音频时长预算：所有槽位时长相加，并按最慢
变速因子预留长度。超预算的单条样本会报错，需要调整来源过滤或 batch 预算。
`batching.max_samples` 可同时限制条数；设置时长预算后，不再用
`per_device_train_batch_size` 决定每批条数。该预算不包含文本 token 或输出生成长度，
实际显存仍受文本长度和模型影响。

`num_buckets` 控制时长桶数量；`buffer_size` 控制交织流的分桶窗口。
同一训练 batch 始终保持相同音频槽位数，collator 补齐时间轴。
不设置时长预算时，使用固定条数上限，仍按槽位分组。

所有 rank 使用同一个确定性计划，再分配完整 batch，避免二次分片。
末尾 batch 数不能整除 world size 时，默认重复开头的完整 batch 补齐；
`batching.drop_last: true` 则丢弃不足一组 rank 的尾部 batch。因此实际出现次数可能
比基础配额略多或略少。它不表示丢弃所有不足 max_samples 的 batch。

SFT 必须显式设置正数 `MAX_STEPS`，因为每轮 batch 数可以随时长和轮换变化。
固定条数推算出的 Trainer samples/s 不适合作为动态 batch 的真实音频吞吐量。
验证集仍使用 Trainer 的固定条数组批，关闭增强与随机热词；验证混用不同槽位数时
使用 `per_device_eval_batch_size=1`，或分开验证。

## 在线增强

所有增强默认关闭。波形依次进行变速、RIR 卷积和按 SNR 混噪；SpecAugment 在
提取 mel 特征后执行，不改变有效长度。开启噪声、混响的配置例如：

```yaml
augmentation:
  noise_prob: 0.3
  noise_snr_db: [5, 20]
  rir_prob: 0.2
```

启用时必须同时配置顶层 `noise_sources` / `rir_sources`，每项同样引用 Catalog 的
`dataset_id`、`version`、`split`。资源支持 cuts 或 recordings。
训练随机性包含 seed、epoch、样本索引及重复位置，同样的采样位置可复现，不同
reps 可以获得不同增强。索引自带 epoch，不受 persistent workers 预取跨 epoch 影响。

变速后的单条音频仍受模型窗口限制，应根据最慢速度因子设置来源的 max_duration，
避免音频截断而标签未截断。

## 断点恢复

每个 SFT checkpoint 写入 `catalog_sampler.json`，保存实际数据 epoch、已经完成训练的
batch 游标和配置签名。预取到但尚未训练的 batch 不计入游标。
恢复时确定性重建采样计划，从下一条未训练 batch 继续；同时由 Trainer 恢复模型、
优化器、调度器及 RNG。不要仅用 global_step 推算数据位置。

```bash
# 保持原来的 DATA_CONFIG、batch 参数和 GPU 数量。
RESUME_FROM_CHECKPOINT=/path/to/checkpoint-200 \
MODEL=/path/to/open-audio-llm-hf-checkpoint \
DATA_CONFIG=examples/configs/data/catalog_mixed.yaml \
CUDA_VISIBLE_DEVICES=2,3 NPROC_PER_NODE=2 MAX_STEPS=1000 \
bash examples/train/sft/train.sh
```

数据配方、记录顺序、batch 参数或 world size 改变会拒绝恢复。旧 checkpoint 没有
采样器状态时，应作为新训练的模型初始化，不宣称恢复了原数据位置。

## 适用范围

上述混合 / 动态 batch 接入的是 **SFT 数据并行训练**。GRPO 继续支持 YAML 数据源和
在线波形增强，保留其原生 generation-group 采样器；配置混合参数或动态 batch 会明确
报错，避免破坏同一 prompt 的多次生成分组。SpecAugment 仍仅限 SFT。
Sequence parallel / tensor parallel 未接入这一 batch sampler。

目前 cut 元数据、样本事实和本轮索引计划仍在内存中，波形按需读取；不具备 icefall
预切物理 shard 的按片懒加载能力。入口不使用 HF dataset.map 预编码、packing 或
padding_free。旧离线转换和索引生成脚本已移除。
