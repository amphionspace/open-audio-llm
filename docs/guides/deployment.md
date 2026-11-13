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

用说字写法 think 训练过的模型（[训练指南](training.md)）要在推理时开 think，就在 serve YAML 的 `task.arguments` 加
`--chat-template: {path: <仓库>/src/open_audio_llm/integrations/vllm/qwen3_asr_think.jinja}`，请求带 `chat_template_kwargs.enable_thinking: true`。
- 不带该参数的请求与原模板逐字相同，所以普通 ASR 和 TS-ASR 服务都可以加这个参数。
- 请求和输出格式见[评测协议](evaluation.md)。
- 上线前还要加一道校验：think 里的原话和写作的字都要出现在转写里，不满足就改用 think 关重解。这一步尚未实现。

### 目标说话人 ASR（TS-ASR）

TS-ASR 使用独立服务配置 [serve/tsasr.yaml](../../examples/configs/serve/tsasr.yaml)：在注册音和混合音之间插入可学习 `[SEP]`，音频注意力使用全局窗口 `n_window_infer=1000000000`。普通 ASR 配置保持 SEP 关闭。需要分块注意力时，在 TS-ASR YAML 中设置 `COT_AUDIO_CHUNKED_ATTN: '1'` 并移除 `--hf-overrides`。实现位于 `src/open_audio_llm/tsasr/`。checkpoint-34479 的服务方式见 [评测结果](../models/ckpt34479-evaluation.md)。

TS-ASR 原生 PyTorch 推理与 vLLM 0.18 都在提取 Mel 前将拼接波形末尾补零到
`hop_length` 的整数倍（16 kHz 下为 160 samples，最多补 159 samples），再分别处理注册音与混合音。
服务配置关闭 prefix cache：BF16 下，同一请求的冷、热缓存计算路径可能改变
`Chinese/None` 的出字判断。关闭后会失去重复前缀的计算复用；它不保证不同 batch、
attention kernel 或 PyTorch/vLLM 之间逐 token 相同，`temperature=0` 也不提供这种保证。
每条请求可以带自己的注册时长：请求体加 `"mm_processor_kwargs": {"enroll_seconds": <秒>}`，
音频仍是「注册 + 混合」拼成的一段 wav，服务在该秒数处切开，两段各自提取 Mel（各自最长 30 秒）。
注册音要先补零到 160 samples 的整数倍（最多 10 ms，不改变其 Mel），否则返回 400。
不带该参数时按服务端 `AMPHION_TSASR_ENROLL_SEC` 切分，旧客户端行为不变（34479 上 20/20 条输出与旧请求逐字一致）。
旧版独立交付包包含源码副本，需要更新其 TS-ASR runtime，并在启动参数中加入
`--no-enable-prefix-caching`；仅更新本仓库不会修改已运行的服务。

### 普通 ASR 置信度拒识（可选中间件）

**作用**：远处、听不清的背景人声会被模型转写成不稳定的文字。
[`confidence_gate.py`](../../src/open_audio_llm/integrations/vllm/confidence_gate.py)
是 vLLM 的 ASGI 中间件，满足下面任一条件时，把这次识别改成模型自己的「无语音」输出
`language None<asr_text>`：

- 转写开头 16 个 token 的平均 logprob 低于阈值；
- 输出撞到长度上限，并且是复读。

**来源**：问题 91。标定和评测见
[`2026-10-30-issue91-background-loops`](../experiments/2026-10-30-issue91-background-loops/README.md)。

**启用方式**：只挂在普通 ASR 服务上；TS-ASR 有自己的出字判断，不挂。在服务 YAML 里加两项：

```yaml
runtime:
  environment:
    AMPHION_CONFIDENCE_THRESHOLD: '-0.831'   # 只对 054 avg3 标定，换 checkpoint 要重标
task:
  arguments:
    --middleware: open_audio_llm.integrations.vllm.confidence_gate.ConfidenceGate
```

模块只依赖标准库。交付包可以把它作为顶层模块放进 `runtime/`，然后用
`--middleware confidence_gate.ConfidenceGate` 启用。

**只拦 HTTP `POST /v1/chat/completions`**：进程内直接调 vLLM 引擎的服务不经过这个接口，只换权重或换启动脚本都不会生效，要在解码后实现同样的规则，并在目标硬件上重新标定阈值。

**对客户端的约定**：
- **返回形式不变**：上游 vLLM 统一按非流式解码，拿到完整结果、判定之后，再按客户端要求的形式返回（JSON 或 SSE）。流式客户端拿到的 chunk 不变，只是在解码结束后一次性发出，依赖逐块检测复读的客户端要注意。
- **拒识时以 stop 结束**：被拒时 `finish_reason` 一律为 `stop`；撞到上限的复读也一样，不会再触发按 `length` 判定的重解码。
- **被拒后不要重解码**：被拒的内容是空的，会把「空结果就切半重解码」的客户端逻辑触发起来，每一半单独判定，可能漏过，还多两次调用。客户端要先看 `amphion_confidence.rejected`。
- **中间结果也会被判定**：插件分不清中间结果和定稿，短音频更容易被拒，字幕会变晚或闪动，数字见实验记录第十节。
- **判定信息**：每个响应带 `amphion_confidence`，内容是分数、阈值、规则、是否拒识。只有客户端自己请求了 logprobs，响应里才有 logprobs。
- **带前缀续写**（`continue_final_message`）：
  - 前缀部分用 prompt logprobs 打分，所以前缀必须按 token 截断（audiollm-go 的 `partialPrefix` 就是这样做的）。按字截断会产生模型从不生成的 token 合并，分数会偏低。
  - 这类请求被拒时返回空续写，客户端看到 `amphion_confidence.rejected` 为真时，要自己丢掉前缀。
- **额外开销**：每次请求多算 token logprobs；带前缀的请求还要多算 prompt logprobs。

## Compose 部署

Compose 由配置生成有效文件，默认只检查配置：

```bash
open-audio-llm deploy --config examples/configs/deploy/vllm.yaml --dry-run
open-audio-llm deploy --config examples/configs/deploy/vllm.yaml
```

需要启动时，在部署 YAML 中显式设置 `operation: [up, --build]`。镜像版本、GPU 和模型只读挂载约束沿用原 profile。
