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
cd /path/to/open-audio-llm
pip install -e .
```

vLLM 部署环境必须先固定 vLLM/PyTorch 的 CUDA 版本。不要在空环境里直接安装
未约束的最新版 vLLM；例如 `vllm 0.24` 会解析到 CUDA 13 的 PyTorch wheel，
在只支持 CUDA 12.8 的驱动上会导致 `Failed to infer device type`。

```bash
source /path/to/miniconda3/etc/profile.d/conda.sh
conda create -n vllm python=3.12 -y
conda activate vllm

cd /path/to/open-audio-llm
python -m pip install -U pip
python -m pip install -e ".[vllm]" -c constraints/vllm-cu128.txt
python -m pip check
python - <<'PY'
import torch
import vllm

print("torch", torch.__version__, torch.version.cuda)
print("cuda", torch.cuda.is_available(), torch.cuda.device_count())
print("vllm", vllm.__version__)
PY
```

通过标准：`torch.version.cuda` 为 `12.8`，`torch.cuda.is_available()` 为 `True`，
且 `vllm` 为 `0.17.0`。如果宿主机驱动不是 CUDA 12.8，请先按驱动版本调整
`constraints/vllm-cu128.txt`，再安装。

Qwen3-ASR serving 与 Compose 镜像使用 `constraints/vllm-serving.txt` 中的
vLLM 0.18 已验证线。该路径还需要系统 C 编译器（例如 Debian/Ubuntu 的
`build-essential`），否则 Torch Inductor/Triton 可能在模型 warmup 时失败。

训练/数据/开发环境可以按需安装：

```bash
pip install -e ".[hf,swift,data,dev]"
```

## vLLM 部署

推荐使用仓库提供的单服务部署 profile 构建 vLLM serving 镜像：

```bash
export OPEN_AUDIO_LLM_MODEL=/path/to/qwen3-asr-or-compatible-model
export VLLM_SERVED_MODEL_NAME=amphionasr-1.7b
export VLLM_PORT=8009
# Shared GPU hosts may need a lower value, for example 0.2.
export VLLM_GPU_MEMORY_UTILIZATION=0.9
docker compose -f compose.vllm.yaml up --build
```

该 compose 文件只覆盖本仓库的 vLLM serving 单元；RAG-ASR Triton、
audiollm-server 等多服务编排应继续放在部署仓库里。

启动 OpenAI-compatible vLLM 服务：

```bash
source /path/to/miniconda3/etc/profile.d/conda.sh
conda activate vllm

cd /path/to/open-audio-llm
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
- `-u`：`gpu-memory-utilization`；共享 GPU 上需要低于 vLLM 默认值。
- `-l`：`max-model-len`。
- `-e`：向 vLLM 传 `--enable-mm-embeds`，允许请求传预计算多模态 embedding。
- `-q`：启用 Qwen3-ASR `audio_embeds` embedding bypass 注册。

目标说话人 ASR 与普通 ASR 不要共用同一个服务。普通 ASR 保持
`AMPHION_TSASR_INSERT_SEP` 关闭。TS-ASR 先设置 `AMPHION_TSASR_INSERT_SEP=1`
再启动 `serve.sh`：服务会注册本仓库的 Qwen3-ASR 后端，在注册音频和混合音频之间
插入可学习 `[SEP]`，并把音频注意力改为整段一个窗口
（`n_window_infer=1000000000`）。要恢复配置里的 800 帧窗口，再设置
`COT_AUDIO_CHUNKED_ATTN=1`。checkpoint-34479 的评测数字和三阶段训练配置见
[评测结果](docs/ckpt34479-evaluation.md)与
[三阶段 SFT](docs/qwen3-asr-three-stage-sft.md)。

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

本项目用于 Audio-LLM 后训练，有适用的 clean 训练版本时优先使用，并设置
`require_clean_pass: true` 过滤未通过条目；没有 clean 版本时允许使用原训练集，
不因此阻断 TS、普通 ASR 或热词训练。配方固定版本引用，保留清洗来源的可追溯性。

训练直接读取 `audio-data-contract` Catalog 中的音频清单，取样时构造输入并执行
on-the-fly 增强，不需要离线生成 ShareGPT JSONL 或增强后的 WAV。
先安装本地契约包：`pip install -e /222042021/mingdong/workspace/audio-data-contract`，
再安装训练依赖：`pip install -e ".[swift,data,hf]"`。

通过 `DATA_CONFIG` 选择 YAML 数据配方；SFT 支持来源权重、samples/reps、epoch
分片轮换、按时长与槽位动态组 batch，以及采样位置断点恢复。混合示例见
[catalog_mixed.yaml](examples/configs/data/catalog_mixed.yaml)。启动脚本默认读取相邻 `audio-data-contract`
仓库的 `catalog/` 和本机 `roots.json`，也可用 `AUDIO_DATA_CATALOG`、
`AUDIO_DATA_ROOTS_FILE` 覆盖。配置与增强说明见 [在线训练数据](docs/online_training_data.md)。

smoke SFT：

```bash
set -a
source examples/configs/train/sft.env
set +a
export DATA_CONFIG=examples/configs/data/catalog_smoke.yaml
export MAX_STEPS=2 SAVE_STEPS=2 EVAL_STEPS=2 LOGGING_STEPS=1
bash examples/train/sft/train.sh
```

smoke GRPO：

```bash
set -a
source examples/configs/train/grpo.env
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

- `src/open_audio_llm/`：唯一 Python package。TS-ASR 的 `[SEP]`、全局音频注意力和 vLLM 后端在 `src/open_audio_llm/tsasr/`。
- `src/open_audio_llm/integrations/`：唯一 integrations 实现位置。
- `examples/configs/`：逐项注释的数据 YAML 与训练 env 示例。
- `examples/train/`：统一 SFT / GRPO / rollout 启动入口。
- `examples/model/`、`examples/serve/`、`examples/eval/`：模型转换、推理和评测脚本。
- `docs/`：当前架构与使用说明；`docs/archive/` 保存历史记录。

仓库中不再保留顶层 `src/integrations/`。旧 AmphionASR 的 integrations 能力已经合并到
`src/open_audio_llm/integrations/` 和 `examples/`。

默认路径不依赖 `k2`。Zipformer 仅作为历史 checkpoint 转换或兼容推理的 legacy adapter。

## 文档

- [近期实验记录（2026-09-22–29）](docs/experiments/2026-09-22-to-29.md)：会议 SOT 评测、数据质检、自动合成和 clean/events A/B 状态。
- [checkpoint-34479 评测结果](docs/ckpt34479-evaluation.md)：三阶段 SFT 最终权重的 ASR、热词、TS-ASR 和警务指标。
- [Qwen3-ASR-1.7B 三阶段 SFT](docs/qwen3-asr-three-stage-sft.md)：从底座到 checkpoint-34479 的数据、学习率和冻结配置。

- [TS-ASR 回放与联合训练](examples/train/qwen3-asr/TS_ASR.md)：clean 优先数据、encoder 批处理和中文保持验收。
- `docs/architecture.md`：组件契约和模型组合。
- `docs/data_boundary.md`：离线样本事实与在线训练随机性的边界。
- `docs/migration_from_amphionasr.md`：从 AmphionASR 迁移的边界和归属。
- `docs/archive/project_context.md`：源项目和目标项目上下文。
- `docs/remaining_work.md`：剩余工作和下一步。
- `docs/train_reproduction.md`：当前训练入口与复现步骤。
- `docs/vllm_triton_bypass.md`：vLLM Qwen3-ASR Triton embedding bypass。
- `docs/compatibility_matrix.md`：默认支持和可选路径。
- `docs/legacy_deps.md`：k2 和 Zipformer legacy 依赖策略。
- `docs/self-contained-asr-deployment.md`：本地 Qwen3-ASR 与 AmphionSPEC 模型、
  plugin、镜像及 Kubernetes 部署方法。
