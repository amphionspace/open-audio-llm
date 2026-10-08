# 使用 AmphionEval 评测

本项目负责训练、产出可部署模型和提供推理服务。评测的数据选择、请求构造、输出解析、归一化、打分和报告，都由独立的 [AmphionEval](https://github.com/amphionspace/amphion-eval)（`amphion-eval` 仓库，命令 `ae`）负责。两边只通过下面三份协议交互，**互相不 import 对方的 Python 代码**。

| 组件 | 负责 | 不负责 |
| --- | --- | --- |
| open-audio-llm | 训练、checkpoint 交付、LoRA 合并、vLLM 插件与服务配置、模型专属的输入格式 | 评测数据、参考答案、打分 |
| AmphionEval | 读取 `audio-data-contract` 数据、构造任务请求、调用服务、解析、归一化、打分、报告、W&B 评测指标 | 加载模型权重、ms-swift、vLLM 运行时 |
| 实验层（`open-audio-llm <任务> --config`） | 串联训练 → 合并 → 服务 → 评测，记录执行状态 | 评测算法 |

## 边界协议

### 1. checkpoint 交付（handoff v1）

训练 YAML 的 `task.arguments` 写 `--checkpoint_handoff: {path: "{attempt}/artifacts/checkpoint-handoff.json"}` 后，训练结束时由 rank 0 原子写入：

| 字段 | 含义 |
| --- | --- |
| `framework` | 固定为 `open-audio-llm` |
| `schema_version` | 当前为 `2`；字段含义变化或删除时递增，消费方遇到未知版本应拒绝 |
| `checkpoint` | 交付的 checkpoint 训练时的绝对路径；启用对象存储后本地副本可能已删除 |
| `checkpoint_remote` | 已核验上传的对象存储路径（`bucket:前缀`）；未启用或上传失败时为 `null`，此时 `checkpoint` 仍在本地 |
| `selection` | `final`（要求末步已保存）或 `best`（Trainer 记录的最佳 checkpoint） |
| `global_step` | 训练结束时的步数 |
| `base_model` | 基础模型；`tuner_type` 为 `lora` 时评测前需要先合并 |
| `tuner_type` | `lora` 或 `full` |
| `training_run` | 训练输出目录 |
| `wandb_run_id` | 训练 W&B run ID，用于把评测 run 关联到训练；没有时为 `null` |
| `amphion_eval_version` | 训练前 `ae open-audio-llm check` 通过的 AmphionEval 版本；未配置 `evaluation` 时为 `null` |

v2 相对 v1 的变化：`checkpoint` 不再保证存在于本地，新增 `checkpoint_remote`。不会按目录名猜测"最新" checkpoint。实现见 `src/open_audio_llm/integrations/ms_swift/checkpoint_handoff.py`。

### 2. 推理服务

- 本项目要部署的模型用 vLLM 的 OpenAI 兼容 HTTP 服务，通过 `open-audio-llm serve --config ...` 启动。服务地址、端口和 `--served-model-name` 以 serve YAML 为准：`serve/vllm.yaml` 为 `8009` / `amphionasr-1.7b`，`serve/tsasr.yaml` 为 `8010` / `tsasr`。
- 评测开始前检查 `GET /version`（vLLM 版本）和 `GET /v1/models`（实际加载的模型），并记入评测快照。核验失败直接终止，不换后端。
- 第三方基线按 `AGENTS.md` 的 Inference Backend 规则选择后端，同样以独立进程提供 HTTP 服务或 JSONL worker，并在快照中记录实际后端。

### 3. Qwen3-ASR 系列请求格式

请求发往 `POST /v1/chat/completions`，`temperature=0`。user 消息只放一个 `input_audio`，内容是 16 kHz 单声道 PCM16 WAV 的 base64。

| 任务 | system | 音频 |
| --- | --- | --- |
| 普通 ASR | 空字符串 | 原始音频 |
| 热词 ASR | `Hotwords: w1,w2` | 原始音频 |
| TS-ASR | `ts_prompt.TS_CONCAT_SYSTEM`；有热词时换行追加 `Hotwords: ...` | 注册音频截断或补零到 3 秒，后面紧接混合音频，中间不插静音 |

- TS-ASR 服务必须用 `serve/tsasr.yaml` 启动（`AMPHION_TSASR_INSERT_SEP=1`），由服务端在两段之间插入可学习的 `[SEP]`。拼接的参考实现是 `open_audio_llm.tsasr.concat_audio.pack_ts_transport_b64`。客户端按本节规则自行实现，不 import 这个函数。
- 模型原始输出形如 `language Chinese<asr_text>转写文本`。打分前去掉 `<asr_text>` 前面的语言前缀。

修改以上任何一项（system 文本、拼接时长、输出格式）都属于协议变更：先在本文件更新并说明版本，再同步修改 AmphionEval。

## 使用步骤

### 1. 准备两个环境

模型服务环境（vLLM）按 [安装指南](../get-started/installation.md) 建 conda env `vllm`。评测环境单独建，不装模型运行时：

```bash
source /path/to/miniconda3/etc/profile.d/conda.sh
conda create -n amphion-eval -c conda-forge --override-channels python=3.12 -y
conda activate amphion-eval
python -m pip install /path/to/audio-data-contract
python -m pip install -r requirements-eval.txt \
  --extra-index-url "https://__token__:${GITLAB_TOKEN}@git.amphiondev.com/api/v4/projects/42/packages/pypi/simple"
ae --version   # amphion-eval 0.6.0 (… open-audio-llm handoff schema 2)
```

[requirements-eval.txt](../../requirements-eval.txt) 固定 `amphion-eval[open-audio-llm,legacy-http,tracking]==0.6.0`（GitLab 包仓库项目 42，`GITLAB_TOKEN` 需 read_api，不写进仓库）。`audio-data-contract` 不在包仓库中，需先从其仓库安装。评测配置的 `runtime.pythonpath` 为空：评测进程不 import 本项目。升级 AmphionEval 时同时修改该文件。

如果 conda 报证书错误，先 `export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt`。

### 2. 启动模型服务

```bash
# vllm 环境
open-audio-llm serve --config examples/configs/serve/vllm.yaml    # 普通 / 热词 ASR
open-audio-llm serve --config examples/configs/serve/tsasr.yaml   # TS-ASR
```

LoRA checkpoint 先按 [merge-lora.yaml](../../examples/configs/model/merge-lora.yaml) 合并，再把合并结果填进 serve YAML 的 `task.model`。

### 3. 运行评测

在 [eval/comparison.yaml](../../examples/configs/eval/comparison.yaml) 中：

- `runtime.python` 填评测环境的解释器，例如 `/path/to/miniconda3/envs/amphion-eval/bin/python`。
- `task.server_url` 和 `--port`、`--model` 与服务一致。
- `--test-plan-file` 选择测试集组合，manifest 目录填本机路径。

```bash
open-audio-llm eval --config examples/configs/eval/comparison.yaml --dry-run
open-audio-llm eval --config examples/configs/eval/comparison.yaml
```

启动器先检查服务的 `/version`，再以 `python -m amphion_eval.cli open-audio-llm vllm ...` 运行评测。结果写入执行目录的 `artifacts/comparison/summary.json`，并按配置同步 W&B、核验远端指标。

已有 SOT 预测只需重新打分时：

```bash
open-audio-llm prepare --config examples/configs/prepare/score-sot.yaml
```

### 4. 训练后评测

1. 训练 YAML 加 `--checkpoint_handoff`。
2. 读取 handoff：`tuner_type: lora` 时用 `base_model` 和 `checkpoint` 合并，`full` 时直接使用 `checkpoint`。本地没有 `checkpoint` 时，在合并 YAML 的 `storage.restore` 写入该路径，执行时会从 `checkpoint_remote` 拉回，结束后删除。
3. 同一份权重起两个服务：`serve/vllm.yaml`（ASR、热词、SOT）和 `serve/tsasr.yaml`（TS-ASR，SEP 是服务级开关）。
4. 按 AmphionEval 的 [after-training 配置示例](https://git.amphiondev.com/amphion/amphioneval/-/blob/main/examples/open-audio-llm/after-training.yaml) 写评测 YAML，填 `server_url`/`served_model` 和 `tsasr_server_url`/`tsasr_served_model`，然后运行 `ae open-audio-llm after-training --handoff <handoff.json> --config <评测 YAML>`。缺少所需服务时在推理前失败。

训练前检查：训练 YAML 加

```yaml
evaluation:
  python: {path: /path/to/amphion-eval/bin/python}
  config: {path: after-training.yaml}
```

启动器在训练开始前运行 `ae open-audio-llm check --config ...`（清空 `PYTHONPATH`，用 `tracking.credentials_file` 加载 W&B 凭据）：评测器、服务名与参数、W&B 凭据、发布桶和评测数据的问题一次列出并拒绝训练；AmphionEval 读取的 handoff 版本与训练写出的（2）不一致，或 AmphionEval 低于 0.3.0 时同样拒绝。报告保存为执行目录的 `amphion-eval-check.json`，handoff 记录 `amphion_eval_version`。服务在训练后才启动，评测开始时再探测。

多目标注册 SOT 的已有预测用 `ae open-audio-llm rescore-target-sot` 打分，见 [多目标 SOT](multi-target-sot.md)。

训练中的周期评测使用 `--retention_eval_script`：训练在保存点暂停，以子进程运行该脚本，参数为 checkpoint 路径和输出目录。训练进程不 import 任何评测代码，脚本内部可以调用 `ae`。
