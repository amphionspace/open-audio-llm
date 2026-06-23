# AmphionASR HuggingFace Integration

将 AmphionASR 核心模型实现为 HuggingFace `PreTrainedModel`，支持 `from_pretrained` / `save_pretrained` / PEFT / TRL 等 HF 生态工具链。

## 文件清单

```
src/open_audio_llm/integrations/hf/
  configuration_amphion_asr.py   # AmphionASRConfig（含 audio encoder / projector / text 子配置）
  modeling_amphion_asr.py        # AmphionASRForConditionalGeneration（PreTrainedModel 实现）
  processing_amphion_asr.py      # AmphionASRProcessor（特征提取 + AutoTokenizer）
  zipformer_inference.py         # 自包含 Zipformer 编码器（无 k2/utils 依赖）
  constants.py                   # 多模态特殊 token 常量定义
  __init__.py                    # 导出 Config / Model / Processor
  convert_amphion_to_hf.py       # 训练 checkpoint -> HF 模型目录的转换脚本
  verify_model.py                # 模型 save/load round-trip 冒烟测试
  test_hf_inference.py           # HF 推理端到端测试（单条 + Lhotse 批量）
  test_encoder_wrapper.py        # AudioEncoderWrapper 单元验证
```

## 架构

模型由三个组件组成，通过 `AmphionASRConfig` 统一管理：

```
AmphionASRConfig
  ├── audio_encoder_config
  │     ├── AmphionASRAudioEncoderConfig  (Qwen3 / Qwen3Omni, 128 mel bins)
  │     └── ZipformerAudioEncoderConfig   (Zipformer, 80 mel bins)
  ├── AmphionASRProjectorConfig           (Linear-SwooshR-Linear 多模态投影)
  └── text_config                         (Qwen CausalLM 语言模型)
```

运行时结构：

```
AmphionASRForConditionalGeneration
  ├── audio_encoder         # AudioEncoderWrapper (统一接口)
  │     ├── Qwen3AudioEncoderWrapper    → qwen_asr.Qwen3ASRAudioEncoder
  │     ├── Qwen3OmniAudioEncoderWrapper → qwen_asr.Qwen3OmniMoeEncoder
  │     └── ZipformerAudioEncoderWrapper → zipformer_inference.Zipformer
  ├── multi_modal_projector # Linear → SwooshR → Linear，下采样音频到 LLM 维度
  ├── language_model        # Qwen 系列因果语言模型
  └── prompt_embedding      # 特殊 token 可学习嵌入
```

### 特征提取策略

| 编码器 | 提取方式 | Mel bins | 备注 |
|--------|---------|----------|------|
| Qwen3 / OmniMoe | `WhisperFeatureExtractor` | 128 | Whisper mel scale |
| Zipformer | `torchaudio.compliance.kaldi.fbank` | 80 | Kaldi mel scale，与 lhotse `FbankConfig` 对齐 |

### 设计要点

- **统一编码器接口**：所有编码器类型通过 `AudioEncoderWrapper` 抽象 + `build_audio_encoder` 工厂函数统一，`forward()` 无 if/elif 分支
- **纯 PyTorch SwooshR**：重新实现 `k2.swoosh_r` 激活函数，推理时无 C++ 扩展依赖
- **特殊 token 持久化**：`<speech>` / `<start_text>` 等 token ID 写入 `config.json`，不再需要运行时 patch

转换后的模型目录包含 `configuration_amphion_asr.py`、`modeling_amphion_asr.py`、`processing_amphion_asr.py` 等源文件，通过 `config.json` 中的 `auto_map` 字段实现 `trust_remote_code` 自动发现。

## 使用方式

### 模型转换

使用上层入口脚本将训练 checkpoint 转为 HF 格式：

```bash
bash examples/model/convert_amphionasr_checkpoint.sh -c <训练 .pt checkpoint> -o <输出目录> -- \
  --encoder-weights <encoder .pth 权重> \
  --encoder-config <encoder JSON 配置> \
  --llm-path <基座 LLM 目录 (HF 格式)> \
  --encoder-type qwen3asr
```

或直接调用 Python 脚本指定参数：

```bash
# Qwen3 编码器
python -m open_audio_llm.integrations.hf.convert_amphion_to_hf \
  --amphion-ckpt <训练 .pt checkpoint> \
  --encoder-weights <encoder .pth 权重> \
  --encoder-config <encoder JSON 配置> \
  --llm-path <基座 LLM 目录 (HF 格式)> \
  --encoder-type qwen3asr \
  --output-dir <输出目录>

# Zipformer 编码器（不需要 --encoder-config，使用内置 preset）
python -m open_audio_llm.integrations.hf.convert_amphion_to_hf \
  --amphion-ckpt <训练 .pt checkpoint> \
  --encoder-weights <encoder .pth 权重> \
  --llm-path <基座 LLM 目录 (HF 格式)> \
  --encoder-type zipformer \
  --zipformer-model-type custom_noncausal \
  --ds-rate 4 \
  --output-dir <输出目录>
```

参数说明：

| 参数 | 必填 | 默认值 | 说明 |
|------|------|--------|------|
| `--amphion-ckpt` | 是 | - | 训练 `.pt` checkpoint 路径 |
| `--encoder-weights` | 是 | - | 编码器 `.pth` 权重文件 |
| `--encoder-config` | 条件 | - | 编码器 JSON 配置文件（Zipformer 不需要） |
| `--llm-path` | 是 | - | 基座 LLM 目录（HF 格式） |
| `--encoder-type` | 否 | `qwen3asr` | 编码器类型：`qwen3asr` / `qwen3omni_captioner` / `qwen3omni` / `zipformer` |
| `--zipformer-model-type` | 条件 | - | Zipformer 预设：`custom`（causal）/ `custom_noncausal`（非 causal） |
| `--ds-rate` | 否 | `1` | 投影器下采样率 |
| `--merge-lora` | 否 | `True` | 是否将 LoRA 权重合并到基座模型 |
| `--output-dir` | 是 | - | 输出 HF 模型目录 |

### HF 推理

```bash
# 单条音频
python -m open_audio_llm.integrations.hf.test_hf_inference --audio /path/to/audio.wav
python -m open_audio_llm.integrations.hf.test_hf_inference --audio /path/to/audio.wav --task asr_zh
python -m open_audio_llm.integrations.hf.test_hf_inference --audio /path/to/audio.wav --prompt "Transcribe the following English audio:"

# Lhotse 批量模式
python -m open_audio_llm.integrations.hf.test_hf_inference \
  --recordings recs.jsonl.gz --supervisions sups.jsonl.gz \
  -n 50 --output results.jsonl
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--audio` | (单条必填) | 音频文件路径（wav/flac/mp3） |
| `--model-dir` | `models/Zipformer-noncausal-Qwen2.5-1.5B-Instruct-hf` | HF 模型目录 |
| `--task` | `asr` | 任务类型：`asr` / `asr_en` / `asr_zh` |
| `--prompt` | `None` | 自定义 prompt（覆盖 `--task`） |
| `--max-new-tokens` | `200` | 最大生成 token 数 |
| `--dtype` | `float16` | 推理精度：`float16` / `bfloat16` / `float32` |
| `--recordings` | `None` | Lhotse recordings JSONL 路径（批量模式） |
| `--supervisions` | `None` | Lhotse supervisions JSONL 路径（批量模式） |
| `-n` | `None` | 批量模式最大处理条数 |
| `--output` | `None` | 批量结果输出 JSONL 路径 |

### 模型验证

冒烟测试，无需真实 checkpoint，验证 config round-trip、模型实例化、save/load、PEFT 兼容性：

```bash
python -m open_audio_llm.integrations.hf.verify_model
```

### AudioEncoderWrapper 验证

验证编码器抽象层的正确性（输出长度公式、state_dict 迁移、Zipformer SwooshR 数值一致性）：

```bash
python -m open_audio_llm.integrations.hf.test_encoder_wrapper
```

## 特殊 Token

模型使用以下特殊 token（定义在 `constants.py`）：

| Token | 用途 |
|-------|------|
| `<speech>` | 音频特征占位符，推理时被编码器输出替换 |
| `<start_speech>` | 音频段起始标记 |
| `<end_speech>` | 音频段结束标记 |
| `<start_text>` | 文本段起始标记 |
| `<end_text>` | 文本段结束标记 |
