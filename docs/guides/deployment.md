# 自包含 ASR 模型部署

先在本仓库工作区的以下目录放置完整模型。模型目录由 `.gitignore` 排除，
克隆仓库不会下载权重：

| 服务 | 本地目录 | vLLM architecture | 对外模型名 |
|---|---|---|---|
| Qwen3-ASR | `models/qwen3-asr-1.7b` | `Qwen3ASRForConditionalGeneration` | `Qwen/Qwen3-ASR-1.7B` |
| AmphionSPEC | `models/amphion-spec` | `AmphionASRForConditionalGeneration` | `AmphionSPEC` |

AmphionSPEC 使用兼容 `AmphionASRForConditionalGeneration` 的情感模型导出，
目录应包含权重、tokenizer、processor 和模型侧 Python 文件。

## Plugin 边界

AmphionSPEC 不能只靠 checkpoint 交给原生 vLLM。启动时还必须设置
`OPEN_AUDIO_LLM_ENABLE_LEGACY_AMPHION_ASR=1`，由本仓库的
`vllm.general_plugins` entry point 注册 `AmphionASRForConditionalGeneration`。
自包含镜像已经设置该变量并安装当前仓库，不会在构建时 clone AmphionASR。

Qwen3-ASR 使用 vLLM 0.18 的原生模型实现，不启用 Amphion plugin、热词召回、k2、
forced aligner 或副模型。

## 校验本地模型

```bash
python scripts/verify_deployment_models.py
```

## 构建包含权重的镜像

先构建公共 vLLM runtime，再构建两套包含模型的镜像：

```bash
docker build \
  -f docker/Dockerfile.vllm-serving \
  -t open-audio-llm/vllm-serving:0.18.0 .

docker compose -f compose.asr-models.yaml build
```

公共 runtime 使用 Dockerfile 专属 `.dockerignore` 排除 `models/`，不会把两份权重
重复打入基础层；两个 bundle 镜像各自只复制自己需要的模型。

模型在镜像构建时从仓库的 `models/` 目录复制进去，因此 Pod 启动时不下载模型，也不
需要模型 URL、SHA256 或 Hugging Face token。代价是镜像较大；向远端集群部署时需要
把镜像推到该集群可访问的 registry，或者通过集群运行时的本地镜像导入命令载入节点。

本机启动：

```bash
docker compose -f compose.asr-models.yaml up
```

- Qwen3-ASR：`http://127.0.0.1:8011`
- AmphionSPEC：`http://127.0.0.1:9001`

Compose 端口仅绑定本机回环地址，服务未配置 API 鉴权。需要远程访问时，先配置
带鉴权的代理或受控网络入口，再显式调整端口发布；Kubernetes Service 默认为集群内访问。

## Kubernetes

`deploy/k8s/asr-models.yaml` 使用镜像内模型，不创建 PVC 或 initContainer：

```bash
kubectl apply -f deploy/k8s/asr-models.yaml
kubectl -n audiollm rollout status deployment/qwen3-asr --timeout=15m
kubectl -n audiollm rollout status deployment/amphion-spec --timeout=15m
```

远端 registry 部署前应把清单中的两个 `image` 改成实际镜像地址和不可变 tag/digest。
默认每个模型申请一张 GPU；共享同一张 GPU 需要由集群显式配置 device-plugin
time-slicing，清单不会假设 GPU 能隐式共享。

## 配置化 vLLM 服务

本仓库的 vLLM 服务都从 YAML 启动，无需手工 export：

```bash
open-audio-llm serve --config examples/configs/serve/vllm.yaml
open-audio-llm serve --config examples/configs/serve/tsasr.yaml
```

Qwen3-ASR embedding bypass 在 `serve/vllm.yaml` 中通过 `--enable-mm-embeds` 和对应插件运行时设置开启，仍使用根项目的 `vllm.general_plugins`，无需单独安装插件子目录。原理见 [vLLM Triton Audio Embedding Bypass](../reference/vllm-triton-bypass.md)。

### 目标说话人 ASR（TS-ASR）

TS-ASR 使用独立服务配置 [serve/tsasr.yaml](../../examples/configs/serve/tsasr.yaml)：在注册音和混合音之间插入可学习 `[SEP]`，音频注意力使用全局窗口 `n_window_infer=1000000000`。普通 ASR 配置保持 SEP 关闭。需要分块注意力时，在 TS-ASR YAML 中设置 `COT_AUDIO_CHUNKED_ATTN: '1'` 并移除 `--hf-overrides`。实现位于 `src/open_audio_llm/tsasr/`。checkpoint-34479 的服务方式见 [评测结果](../models/ckpt34479-evaluation.md)。

## Compose 部署

Compose 由配置生成有效文件，默认只检查配置：

```bash
open-audio-llm deploy --config examples/configs/deploy/vllm.yaml --dry-run
open-audio-llm deploy --config examples/configs/deploy/vllm.yaml
```

需要启动时，在部署 YAML 中显式设置 `operation: [up, --build]`。镜像版本、GPU 和模型只读挂载约束沿用原 profile。
