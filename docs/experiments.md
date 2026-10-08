# 实验目录与配置规范

已有实验的结果、轨迹、原始证据及对象存储权重见 [2026-10-06 归档](experiments/2026-10-06/README.md)、[2026-10-08 归档](experiments/2026-10-08/README.md)。归档是已有产物的只读快照，不是新训练任务。

## 自我迭代流程

每一轮训练或评测都要能回答三个问题：为什么做、结果如何、下一轮因此改什么。

1. **启动前**：在任务 README 写明要验证的假设、相对上一轮改变的唯一或主要变量、停止条件。配置只改这些变量；数据配置的增删（如 `exclude_records`）随执行冻结。
2. **训练中**：W&B 同步并核验远端；训练内评测只用选模集，不用对外比较的评测集；核验实际生效的学习率、可训练参数和数据游标，不以配置为准。
3. **训练后标准评测**：对候选 checkpoint 跑两套固定评测并与基线同口径比较：
   - 固定集（338 条，`evaluate-a800`）：与历史结果和 MOSS 15.53%（官方 vLLM）对比；
   - 180 秒会议评测集（`evaluate-meeting-benchmark` → `score-meeting-benchmark`）：中文、英文、6/8/10 人和复读条数；MOSS 输出已缓存，复用即可。
   新 checkpoint 只需加入 `models`，已完成的推理通过 `completed_inference` 复用。
4. **归档**：执行记录、原始输出、逐条打分和保留的 checkpoint 用 `scripts/stage_experiment_upload.py` 暂存后 `ab push` 到 `whai:open-audio-llm/runs/<实验>/`，路径与本地一致；清单和排除规则提交到 `docs/experiments/<日期>/`。保留：每次训练最后可续训的 checkpoint 和选中的最佳点；合并模型、已确认退化的中间点可不传，但须在清单中写明。
5. **结论**：阶段结束时写 `docs/experiments/<日期>/README.md`（结论先行，执行表含做了什么、结果、决定）和 `results.json`（机器可读的关键指标）。失败或中止的执行同样记录原因，不删除。

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

训练 checkpoint 和执行记录同步到对象存储 `whai:open-audio-llm`（SenseCore AOSS，内网），远端路径与 `storage.local_root` 下的本地路径一致，例如 `whai:open-audio-llm/runs/<实验>/tasks/<任务>/attempts/<编号>/`。在任务 YAML 写 `storage`（`executable` 指向 AmphionBucket 的 `ab`、`remote`、`local_root`）即启用，示例训练、合并和评测配置默认开启：

- 训练：每次保存后由 rank 0 在后台上传权重，中间 checkpoint 不传优化器、调度器、RNG 和 DeepSpeed `global_step*` 状态；训练结束时最后一个 checkpoint 连同这些状态完整上传。`ab` 核验全部文件后才删除本地副本；训练中保留最新和最佳 checkpoint 供中断续训和 `load_best_model_at_end`。本地清理由上传器负责，启用时不能设置 `save_total_limit`。上传记录写在训练输出目录的 `storage-uploads.json`，失败的 checkpoint 保留在本地。
- 交付位置：配置了 `--checkpoint_handoff` 的训练结束后，交付 checkpoint 的 `checkpoint_remote`（未上传时为本地路径）写入 `status.json` 的 `checkpoint_handoff`，并显示在执行 README 的状态块中。
- 执行结束：整个执行记录（配置、状态、日志、评测结果等）以覆盖方式镜像到远端，结果写入 `status.json` 的 `storage`。同步失败不改变执行退出码。
- 输入：`storage.restore` 列出的路径若本地不存在，执行前从远端拉回，结束后删除。实验评测按 producer 选中的训练 checkpoint 自动加入该列表。
- 需要 AmphionBucket ≥ 0.5.0：从 GitLab 包仓库（项目 34）安装到独立环境，例如 `uv tool install amphionbucket==0.5.0 --index-url <包仓库>`。凭证在运行时读取，放在 `~/.config/amphion-bucket/credentials.ini`（权限 600）或 `AMPHION_BUCKET_<PROFILE>_ACCESS_KEY/SECRET_KEY`，不写进 YAML；`ab credentials` 可查看来源。
- 启动前检查：任务开始前确认 `ab` 可执行、版本 ≥ 0.5.0、目标桶有凭证、执行目录位于 `local_root` 内，任何一项不满足都在训练前失败，不等到第一次上传。
- 失败定位：任何上传或拉取失败都会自动按顺序检查：本地路径、`ab` 是否可执行且支持 `--json`、是否配置该桶、该桶是否有凭证、远端冲突或未确认文件、桶能否访问、远端数据是否存在，得出 `cause`、说明和 `ab` 原始输出。原因写入 `storage-uploads.json`、`status.json` 的 `storage`、执行 README 和结束输出，并打印到 stderr。
- 补传：`open-audio-llm storage sync --config <任务 YAML> --attempt <执行目录>` 重新上传指定执行，并在核验后删除其中的本地 checkpoint；执行仍持有锁时拒绝。

本项目要部署的 checkpoint 用 vLLM 推理、评测和比较，启动前检查配置后端与目标解释器中的 vLLM；HTTP 评测检查服务 `/version`，客户端无需安装模型运行时。示例实验核验实际引擎 `backend` 和类来源并保存 runtime 证据。第三方基线（含 AntSpeaker）使用官方推荐推理方式并固定版本，在独立进程中运行，后端记入快照；存疑项隔离，不恢复人工听审。

Compose 使用 `open-audio-llm deploy --config examples/configs/deploy/vllm.yaml --dry-run` 预览，去掉 `--dry-run` 后生成有效文件并执行 YAML 中的 `operation`，示例默认只做 `config` 检查。需要部署时在配置写明 `operation: [up, --build]`。模型只读挂载、GPU 约束、绑定地址和镜像版本沿用原 profile；容器读取独立服务 YAML。vLLM serving 仍为 constraints 固定的 0.18.0，常规 vLLM 依赖仍为 0.17.0；没有构建或启动镜像验证，需在对应 CUDA/驱动环境部署验证。

本地旧实验迁移工具 `scripts/migrate_experiments.py` 先预览路径映射，`--apply` 在进程与文件占用检查通过后重命名，并按文件大小/inode 和符号链接核验；不复制权重。`scripts/configure_clean_events.py --root ... --write` 从保存的实际训练参数生成配置及导航，拒绝覆盖已有不同内容。原始脚本保存在 `shared/legacy-scripts`，当前脚本从生效配置获取实验路径。

迁入执行的 `effective.yaml` 标注 `recording.reconstructed`：它按原始参数和路径映射重建，不能当作当时保存的原始 YAML。失败评测可显式选择已有记录恢复；其余任务重试创建新编号。`scripts/verify_experiment_layout.py --root runs/clean-events-ab-20260928` 核对原文件、链接、关键训练清单和配置预览。Git 只保存概览、配置、执行元数据和当前脚本，音频、权重与原始产物保留本地。

示例实验迁移了 1,096 个文件和链接，共 99,143,088,283 字节；两组已有 1000 步完成证据，固定评测失败。历史状态中的冲突保留并标注待核实，不改训练配方、数据版本、评测集合或原反馈。
