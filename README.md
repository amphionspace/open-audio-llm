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

## 配置与运行

正式入口只接受配置路径、任务选择、恢复和预览。先填写 YAML 中的模型、解释器、GPU、端口与数据路径；相对路径以所属配置文件为基准。

```bash
open-audio-llm train --config examples/configs/train/sft.yaml --dry-run
open-audio-llm train --config examples/configs/train/sft.yaml
open-audio-llm train --config examples/configs/train/grpo.yaml
open-audio-llm serve --config examples/configs/serve/vllm.yaml
open-audio-llm eval --config examples/configs/eval/comparison.yaml
open-audio-llm model --config examples/configs/model/convert-legacy.yaml
```

训练、数据准备、合成、评测、服务、rollout 和模型处理配置见 [examples/configs](examples/configs/README.md)。shell 文件是薄入口，不再接受 `MODEL`、`MAX_STEPS`、`OUTPUT_DIR` 等业务环境变量。数据加载器只从 YAML 读取 Catalog、roots 和缓存路径。

训练和评测必须同步 W&B，并核验远端 run URL 和实际指标；记录保存在每次执行目录。凭据留在环境中。推理与模型对比使用 vLLM，AntSpeaker 声纹核验使用已授权的官方 PyTorch；不会自动回退后端。

Compose 由配置生成有效文件，默认只检查配置：

```bash
open-audio-llm deploy --config examples/configs/deploy/vllm.yaml --dry-run
open-audio-llm deploy --config examples/configs/deploy/vllm.yaml
```

需要启动时，在部署 YAML 中显式设置 `operation: [up, --build]`。镜像版本、GPU 和模型只读挂载约束沿用原 profile。Qwen3-ASR embedding bypass 在 `serve/vllm.yaml` 中通过 `--enable-mm-embeds` 和对应插件运行时设置开启，仍使用根项目的 `vllm.general_plugins`，无需单独安装插件子目录。

## 实验目录

每个实验根目录只放概览、方案和目录；每项任务提供中文输入、完成条件、进展和证据，失败与补充执行归入原任务的 `attempts`，完成状态与质量结论分开记录。

```bash
open-audio-llm experiment run --config runs/clean-events-ab-20260928/experiment.yaml --dry-run
open-audio-llm experiment run --config runs/clean-events-ab-20260928/experiment.yaml --task evaluate
```

示例实验两组已完成 1000 步，固定评测失败，尚无最终对比结论；迁移保留原始数据、日志、失败版本与反馈。其他历史实验只补导航，产物保留原位。规范见 [实验目录与配置](docs/experiments.md)。

训练直接读取 `audio-data-contract` Catalog，在取样时在线增强，不生成离线 ShareGPT。适用 clean 训练版本优先使用并设置 `require_clean_pass: true`；没有 clean 版本时允许原训练集。固定数据版本与真实开发/测试标签，禁止为通过门槛删难例。

## 目录边界

- `src/open_audio_llm/`：唯一 Python package。
- `src/open_audio_llm/integrations/`：唯一 integrations 实现位置。
- `examples/configs/`：显式的数据与运行 YAML。
- `examples/train/`：统一 SFT / GRPO / rollout 启动入口。
- `examples/model/`、`examples/serve/`、`examples/eval/`：模型转换、推理和评测脚本。
- `docs/`：当前架构与使用说明；`docs/archive/` 保存历史记录。

仓库中不再保留顶层 `src/integrations/`。旧 AmphionASR 的 integrations 能力已经合并到
`src/open_audio_llm/integrations/` 和 `examples/`。

默认路径不依赖 `k2`。Zipformer 仅作为历史 checkpoint 转换或兼容推理的 legacy adapter。

## 文档

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
