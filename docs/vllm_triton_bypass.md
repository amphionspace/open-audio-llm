# vLLM Triton Audio Embedding Bypass

## 问题复述

目标是让 vLLM 服务直接消费 RAG-ASR Triton 返回的 post-projector audio
embedding，跳过 vLLM 内部 Qwen3-ASR audio encoder，并用同一条请求验证 raw
audio 路径与 Triton embedding 路径输出一致。

## 关键假设

- 高风险：vLLM 原生 `Qwen3ASRForConditionalGeneration` 不完整支持
  `audio_embeds`，需要目标仓库 plugin 覆盖注册。
- 中风险：Triton `PROJECTOR_OUT` 必须已经是 LLM hidden size 维度的
  post-projector frames。当前 Amphion-1.7B 验证样本为 `(T, 2048)`。
- 低风险：验证客户端可在 `triton` 环境运行；该环境已有 `tritonclient`、
  `soundfile` 和 vLLM 的 tensor serialization helper。

## 实现入口

- `src/open_audio_llm/integrations/vllm/plugin/qwen3_asr_embeds.py`
  为 Qwen3-ASR 增加 `audio_embeds` parser、placeholder 长度、MRoPE 位置
  fallback 和模型侧 embedding 直通。
- `src/open_audio_llm/integrations/vllm/plugin/__init__.py`
  在 `OPEN_AUDIO_LLM_ENABLE_QWEN3_ASR_EMBEDS=1` 时覆盖注册
  `Qwen3ASRForConditionalGeneration`。
- `src/open_audio_llm/integrations/vllm/triton_audio_embed.py`
  提供 Triton HTTP client 和 vLLM `audio_embeds` content block 构造。
- `examples/serve/vllm/serve.sh`
  提供 `-e` 与 `-q` 服务启动开关。
- `examples/serve/vllm/validate_triton_bypass.py`
  用同一条音频分别请求 raw `input_audio` 和 Triton `audio_embeds`。

## 安装

在 vLLM 服务环境安装目标仓库，确保 vLLM 能发现 `vllm.general_plugins`
entry point。服务环境必须固定 vLLM/PyTorch 的 CUDA 版本；在 CUDA 12.8
驱动机器上不要直接安装未约束的最新版 vLLM，否则可能解析到 CUDA 13 的
PyTorch wheel 并在启动时报 `Failed to infer device type`。

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

上面的 `.[vllm]` 路径保留远端 CUDA 12.8 通用约束线。Qwen3-ASR serving 和
Compose 镜像使用 `constraints/vllm-serving.txt` 中的 vLLM 0.18 已验证线。
constraints 文件只写 canonical
package pins，不写 extras；需要 HTTP client 时在 install 命令中请求
`tritonclient[http]`。运行时镜像还必须包含 C 编译器，例如
`build-essential`，否则 Torch Inductor/Triton 可能在模型 warmup 阶段报
`Failed to find C compiler`。

如果要构建可复现镜像，可使用单服务 compose profile：

```bash
export OPEN_AUDIO_LLM_MODEL=/path/to/qwen3-asr-or-compatible-model
export VLLM_SERVED_MODEL_NAME=amphionasr-1.7b
export VLLM_PORT=8009
docker compose -f compose.vllm.yaml up --build
```

对应 Dockerfile 会先用可续传下载预取大 wheel，再按
`constraints/vllm-serving.txt` 安装，避免 pip 在构建时解析到未来的
vLLM、Torch、Transformers 或 CUDA wheel。

如果只是运行验证客户端，也可以使用已有 `triton` 环境：

```bash
source /path/to/miniconda3/etc/profile.d/conda.sh
conda activate triton
cd /path/to/open-audio-llm
python -m pip install -e .
```

## 启动服务

```bash
source /path/to/miniconda3/etc/profile.d/conda.sh
conda activate vllm

cd /path/to/open-audio-llm
bash examples/serve/vllm/serve.sh \
  -m /chenmingjie/lx/RAG-ASR/checkpoints/base/amphion_1.7b_merged \
  -n amphionasr-1.7b \
  -p 8009 \
  -g 2 \
  -t 1 \
  -e \
  -q
```

启动日志应出现类似信息：

```text
Model architecture Qwen3ASRForConditionalGeneration is already registered,
and will be overwritten by the new model class
open_audio_llm.integrations.vllm.plugin.qwen3_asr_embeds:Qwen3ASRForVLLMWithEmbeds.
```

## 健康检查

```bash
python - <<'PY'
import urllib.request
for name, url in [
    ("vllm_health", "http://localhost:8009/health"),
    ("vllm_models", "http://localhost:8009/v1/models"),
    ("triton_ready", "http://localhost:8000/v2/health/ready"),
]:
    with urllib.request.urlopen(url, timeout=5) as r:
        body = r.read().decode("utf-8", errors="replace")
    print(name, "OK", body[:300])
PY
```

## 最小验证

```bash
source /path/to/miniconda3/etc/profile.d/conda.sh
conda activate triton
cd /path/to/open-audio-llm

python examples/serve/vllm/validate_triton_bypass.py \
  --base-url http://localhost:8009 \
  --model amphionasr-1.7b \
  --audio /chenmingjie/mingdong/data/opensource/Libri2Mix/wav16k/min/test/mix_both/4077-13754-0001_5142-33396-0065.wav \
  --enrollment /chenmingjie/mingdong/data/opensource/LibriSpeech/test-clean/4077/13751/4077-13751-0001.flac \
  --triton-url localhost:8000 \
  --triton-model rag_asr_retrieve \
  --triton-top-k 0
```

通过标准：

- `raw_text` 与 `triton_text` 完全一致。
- `exact_match` 为 `true`。
- `triton_embeds` 中每条音频有 `frames_shape`，例如 `[49, 2048]`。

## 常见失败与根因

- `Transformers does not recognize model_type qwen3_asr`：vLLM 版本过旧；
  使用 `constraints/vllm-serving.txt` 中的 vLLM 0.18 已验证线。
- `Failed to find C compiler`：运行时缺少 C 编译器；镜像需要安装
  `build-essential` 或等价包。
- `You must set --enable-mm-embeds`：服务启动时没有传 `-e`。
- `Failed to apply Qwen3ASRProcessor`：plugin 没有覆盖 Qwen3-ASR 原生类，
  通常是没有安装目标仓库、entry point 没加载，或没有传 `-q`。
- `KeyError: audio_feature_lengths`：请求已进入模型侧，但仍在用原生 MRoPE
  逻辑；确认日志中覆盖注册的是
  `open_audio_llm.integrations.vllm.plugin.qwen3_asr_embeds:Qwen3ASRForVLLMWithEmbeds`。
- `At most N audio(s) may be provided`：客户端把单条 audio embedding 传成
  3D batch。正确 content block 是一个 2D tensor，shape 为 `(T, hidden_size)`。

## 下一步精度验证

最小验证只证明协议和模型路径可用。要做精度结论，应在同一服务、同一
Triton hotword pool、相同 prompt 和 `temperature=0` 下扩大样本数，对比
baseline raw audio 与 Triton bypass 的 transcript、归一化文本、WER/CER 和
失败率。
