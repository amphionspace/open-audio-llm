# 安装

所有插件和脚本都已经合并到根项目。只需要在仓库根目录安装，不要进入
`src/open_audio_llm/integrations/vllm/plugin/` 单独执行 `pip install -e .`。

## 环境与 extras

| 环境 | 安装命令 | 用途 |
| --- | --- | --- |
| 基础 | `pip install -e .` | 模型组件与 CLI |
| 训练 / 数据 / 开发 | `pip install -e ".[hf,swift,data,dev]"` | ms-swift SFT/GRPO、Catalog 在线数据、测试 |
| vLLM 推理 | `pip install -e ".[vllm]" -c constraints/vllm-cu128.txt` | 评测与对比用的 vLLM 0.17 |
| vLLM serving | 见下文“vLLM serving 环境”，约束 `constraints/vllm-serving.txt`；镜像按 `docker/Dockerfile.vllm-serving` 构建 | 交付包、`open-audio-llm serve`、Compose 镜像与部署模型评测（vLLM 0.32，CUDA 12.9） |
| 实验跟踪 | `pip install -e ".[tracking]"` | W&B 同步 |

## vLLM 推理环境

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

## vLLM serving 环境

交付包、`open-audio-llm serve`、Compose 镜像和要部署模型的评测都使用
`constraints/vllm-serving.txt` 中的 vLLM 0.32 已验证线（torch 2.13、Transformers 5.19）。
用 CUDA 12.9 构建：NVIDIA 驱动 570 起可用；PyPI 默认的 vLLM/torch 是 CUDA 13，
要求驱动 580 起，所以安装时必须带下面的索引和约束。

```bash
conda create -n vllm-serving-032 python=3.11 -y
conda activate vllm-serving-032
cd /path/to/open-audio-llm
python -m pip install -U pip
python -m pip install -e ".[vllm-serving]" -c constraints/vllm-serving.txt \
  --extra-index-url https://wheels.vllm.ai/0.32.0/cu129 \
  --extra-index-url https://download.pytorch.org/whl/cu129 \
  --extra-index-url https://download.pytorch.org/whl/cpu
python -c 'import torch, vllm.model_executor.models.qwen3_asr; print(torch.__version__, torch.version.cuda, vllm.__version__)'
```

通过标准：输出 `2.13.0+cu129 12.9 0.32.0`。`torchcodec` 固定为 CPU 构建
（`0.17.0+cpu`）：PyPI 版链接 CUDA 13 的库，会让 `import vllm` 在本环境失败，
而 0.17 没有 CUDA 12.9 构建；没有 ffmpeg 时 vLLM 改用 soundfile 解码音频。
该路径还需要系统 C 编译器（例如 Debian/Ubuntu 的 `build-essential`），否则
Torch Inductor/Triton 可能在模型 warmup 时失败。

GRPO 的训练进程内 rollout 仍在训练环境里用 vLLM 0.18（见
[训练入口](../guides/training.md)），两边共用的插件代码（`qwen3_asr_embeds.py`）兼容两个版本。

## 评测环境

评测由独立的 AmphionEval 提供，评测环境不加载模型权重，单独建环境，见
[使用 AmphionEval 评测](../guides/evaluation.md)。

## 下一步

- 填写 [examples/configs](../../examples/configs/README.md) 中的 YAML，先 `--dry-run` 预览。
- 训练入口见 [训练入口与复现](../guides/training.md)，部署见 [部署](../guides/deployment.md)。
