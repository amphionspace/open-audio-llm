# 实验目录与配置规范

每个新实验根目录只放中文概览、实验方案和目录。任务输入、完成条件、进展、结果与证据链接写在任务 README；逐次执行细节放到 attempts，执行完成与质量达标分开记录。

```text
runs/<实验名>/
  README.md
  experiment.yaml
  configs/
  scripts/
  tasks/<序号-任务名>/README.md
  tasks/<序号-任务名>/attempts/<编号>/
    effective.yaml
    status.json
    README.md
    logs/
    artifacts/
  shared/
  tracking/
```

合成补充和失败重试属于原任务的新执行或明确的恢复阶段。保留失败产物、已确认反馈、原始来源证明与代码快照，不覆盖历史结果。其他历史实验只补导航，不搬原产物。

统一入口：

```bash
open-audio-llm experiment run --config runs/clean-events-ab-20260928/experiment.yaml --dry-run
open-audio-llm experiment run --config runs/clean-events-ab-20260928/experiment.yaml --task evaluate
open-audio-llm train --config examples/configs/train/sft.yaml
open-audio-llm eval --config examples/configs/eval/comparison.yaml
open-audio-llm serve --config examples/configs/serve/vllm.yaml
```

YAML 只使用 PyYAML，无多层继承。`runtime` 指定解释器、工作目录、Python 模块搜索路径、GPU、分布式和线程；`data` 引用数据配置；`task` 指定模块/脚本及位置参数、具名参数和无值 flags；`recording` 指定执行目录与结果；`tracking` 指定 W&B 记录。

路径写成 `{path: ...}`，相对所属配置文件解析，`{attempt}` 指当前执行记录。普通字符串如模型 ID 不按路径处理。参数值为布尔时传 `true`/`false`；无值开关放 `task.flags`，需重复出现的参数放 `task.repeated_arguments`。所有任务参数来自 YAML，不支持 CLI 通用覆盖语言。

`--dry-run` 展示解析路径、命令、CPU 预检查和运行时设置，不启动子进程、分配 GPU、创建执行目录或写外部系统。默认跳过已完成实验任务；遇到已有失败记录时必须选择 `--task` 创建新执行，或用 `--task ... --resume <已有执行目录>` 明确恢复。任务必须声明支持恢复；完成执行不允许恢复。PID 文件及目录存在不能证明运行中。

记录器保存开始、结束、实际退出码、配置及结果，转发 SIGINT/SIGTERM 到整个子进程组。生效数据 YAML 会冻结在执行记录中，CPU 预检查与训练使用同一份快照。只有主启动器更新状态和 README，DDP rank 不重复写。

训练与评测必须启用 W&B；CPU 记录器加载配置指定的凭据文件，核验远端 run URL 和实际上传的启动指标后才允许任务启动，结束后核验实际指标与远端状态。凭据不写 YAML、Git 或预览输出。默认 entity/project 为 `1016097967-amphion/open-audio-llm`。启动与结束核验分别保存为 `wandb-start-verification.json`、`wandb-verification.json`。记录进程故障会停止实验，防止实验脱离记录继续运行。

推理、评测和模型比较必须使用 vLLM，启动前检查配置后端与目标解释器中的 vLLM；HTTP 评测检查服务 `/version`，客户端无需安装模型运行时。示例实验核验实际引擎 `backend` 和类来源并保存 runtime 证据。AntSpeaker 使用已授权的官方 PyTorch 后端；存疑项隔离，不恢复人工听审。

Compose 使用 `open-audio-llm deploy --config examples/configs/deploy/vllm.yaml --dry-run` 预览，去掉 `--dry-run` 后生成有效文件并执行 YAML 中的 `operation`，示例默认只做 `config` 检查。需要部署时在配置写明 `operation: [up, --build]`。模型只读挂载、GPU 约束、绑定地址和镜像版本沿用原 profile；容器读取独立服务 YAML。vLLM serving 仍为 constraints 固定的 0.18.0，常规 vLLM 依赖仍为 0.17.0；没有构建或启动镜像验证，需在对应 CUDA/驱动环境部署验证。

本地旧实验迁移工具 `scripts/migrate_experiments.py` 先预览路径映射，`--apply` 在进程与文件占用检查通过后重命名，并按文件大小/inode 和符号链接核验；不复制权重。`scripts/configure_clean_events.py --root ... --write` 从保存的实际训练参数生成配置及导航，拒绝覆盖已有不同内容。原始脚本保存在 `shared/legacy-scripts`，当前脚本从生效配置获取实验路径。

迁入执行的 `effective.yaml` 标注 `recording.reconstructed`：它按原始参数和路径映射重建，不能当作当时保存的原始 YAML。失败评测可显式选择已有记录恢复；其余任务重试创建新编号。`scripts/verify_experiment_layout.py --root runs/clean-events-ab-20260928` 核对原文件、链接、关键训练清单和配置预览。Git 只保存概览、配置、执行元数据和当前脚本，音频、权重与原始产物保留本地。

示例实验迁移了 1,096 个文件和链接，共 99,143,088,283 字节；两组已有 1000 步完成证据，固定评测失败。历史状态中的冲突保留并标注待核实，不改训练配方、数据版本、评测集合或原反馈。
