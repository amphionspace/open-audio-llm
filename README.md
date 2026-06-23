# Open Audio-LLM

`open-audio-llm` 是一个可组合的 Audio-LLM 框架，用来把不同音频编码器接到
Hugging Face CausalLM 上，并统一 Hugging Face、ms-swift 和 vLLM 的训练/部署
路径。

核心结构：

```text
AudioProcessor -> AudioTower -> Connector -> SlotMerger -> CausalLM
```

- `AudioTower`：把音频特征转成 `(B, T, D)` hidden states。
- `Connector`：把音频 hidden size 映射到 LLM hidden size。
- `SlotMerger`：把一个或多个 `<speech>` 槽位替换为连续音频 embedding。
- `CausalLM`：任意支持 `inputs_embeds` 的 Hugging Face causal language model。

## 安装

所有插件和脚本都已经合并到根项目。只需要在仓库根目录安装，不要进入
`src/open_audio_llm/integrations/vllm/plugin/` 单独执行 `pip install -e .`。

基础安装：

```bash
cd /chenmingjie/mingdong/workspace/open-audio-llm
pip install -e .
```

vLLM 部署环境：

```bash
cd /chenmingjie/mingdong/workspace/open-audio-llm
pip install -e ".[vllm]"
```

训练/数据/开发环境可以按需安装：

```bash
pip install -e ".[hf,swift,data,dev]"
```

## vLLM 部署

启动 OpenAI-compatible vLLM 服务：

```bash
cd /chenmingjie/mingdong/workspace/open-audio-llm
bash examples/serve/vllm/serve.sh \
  -m /path/to/model \
  -p 8000 \
  -g 0 \
  -n open-audio-llm \
  -t 1 \
  -d 1 \
  -a 4
```

常用参数：

- `-m`：模型目录，必填；脚本会自动解析 ModelScope 嵌套 `config.json`。
- `-p`：服务端口，默认 `8000`。
- `-g`：`CUDA_VISIBLE_DEVICES`。
- `-n`：`served-model-name`。
- `-t`：tensor parallel size。
- `-d`：data parallel size。
- `-a`：每条 prompt 允许的最大音频数。
- `-e`：向 vLLM 传 `--enable-mm-embeds`，允许请求传预计算多模态 embedding。
- `-q`：启用 Qwen3-ASR `audio_embeds` embedding bypass 注册。

Triton/RAG-ASR embedding bypass 示例：

```bash
bash examples/serve/vllm/serve.sh \
  -m /path/to/qwen3-asr-or-compatible-model \
  -p 8009 \
  -g 0 \
  -n amphionasr-1.7b \
  -e \
  -q
```

`-q` 会在脚本内部设置 `OPEN_AUDIO_LLM_ENABLE_QWEN3_ASR_EMBEDS=1`，并通过根项目
的 `vllm.general_plugins` entry point 注册：

`open_audio_llm.integrations.vllm.plugin:register`

因此不需要、也不应该安装 `src/open_audio_llm/integrations/vllm/plugin/` 子目录。

## 训练与转换

smoke SFT：

```bash
set -a
source configs/train/sft_smoke.env
set +a
bash examples/train/sft/train.sh
```

smoke GRPO：

```bash
set -a
source configs/train/grpo_smoke.env
set +a
bash examples/train/grpo/train.sh
```

legacy checkpoint 转 Open Audio-LLM HF 格式：

```bash
export AMPHION_CKPT=/path/to/legacy.pt
export ENCODER_WEIGHTS=/path/to/encoder.pth
export ENCODER_CONFIG=/path/to/encoder_config.json
export LLM_PATH=/path/to/base_llm
export OUTPUT_DIR=runs/converted_open_audio_llm
bash examples/model/convert_legacy_checkpoint.sh
```

旧 AmphionASR HF 兼容转换入口：

```bash
bash examples/model/convert_amphionasr_checkpoint.sh \
  -c /path/to/amphion_checkpoint.pt \
  -o /path/to/output_hf_dir
```

## 目录边界

- `src/open_audio_llm/`：唯一 Python package。
- `src/open_audio_llm/integrations/`：唯一 integrations 实现位置。
- `examples/`：可运行脚本和配方。
- `docs/`：迁移说明、架构说明和验证记录。

仓库中不再保留顶层 `src/integrations/`。旧 AmphionASR 的 integrations 能力已经合并到
`src/open_audio_llm/integrations/` 和 `examples/`。

默认路径不依赖 `k2`。Zipformer 仅作为历史 checkpoint 转换或兼容推理的 legacy adapter。

## 文档

- `docs/architecture.md`：组件契约和模型组合。
- `docs/data_boundary.md`：离线样本事实与在线训练随机性的边界。
- `docs/migration_from_amphionasr.md`：从 AmphionASR 迁移的边界和归属。
- `docs/project_context.md`：源项目和目标项目上下文。
- `docs/remaining_work.md`：剩余工作和下一步。
- `docs/train_reproduction.md`：训练复现和脚本映射。
- `docs/vllm_triton_bypass.md`：vLLM Qwen3-ASR Triton embedding bypass。
- `docs/compatibility_matrix.md`：默认支持和可选路径。
- `docs/legacy_deps.md`：k2 和 Zipformer legacy 依赖策略。
