# 在 SenseCore ACP 算力池上训练

训练配置加上 `cluster` 段后，`open-audio-llm train --config <YAML>` 不在本机运行，而是用 `sco` 向算力池提交多机任务。执行记录、W&B 和对象存储同步保持原样，由任务中 rank 0 节点的启动器负责。示例见 [sft-acp.yaml](../../examples/configs/train/sft-acp.yaml)。

## 一次性准备

1. 拉取密钥 submodule：`git submodule update --init third_party/amphionkeys`。`sensecore.env` 是公司共享 SenseCore 账号，`sco` 和 AmphionBucket 都使用它。
2. 安装 `sco` 到 `/workspace` 下的目录（这里以 `/workspace/workspace/<用户>/.local-tools/sco` 为例）：

   ```bash
   export SCO_HOME=/workspace/workspace/<用户>/.local-tools/sco
   curl -sSf https://sco.sensecore.cn/registry/install.sh | SCO_DATA_HOME=$SCO_HOME/.data sh
   ```

   registry 只保留最新版本，装好的旧版会让所有 `sco acp` 调用报 `Failed to prepare runtime for the selected region: 404 page not found`（2026-10-16 发生过一次，20260830 → 20260930）。启动器提交前会自动检查并用上面的安装脚本重装；直接用 `sco` 命令查询时遇到这个报错，重新执行上面的安装命令即可，profile 和密钥不受影响。

   然后写入只含区域信息的 profile，不放密钥。启动器提交时把 AmphionKeys 的 key 通过 `SCO_ACCESS_KEY_ID/SECRET` 环境变量传给 `sco`：

   ```bash
   mkdir -p $SCO_HOME/.config/profiles && printf default > $SCO_HOME/.config/active_profile
   cat > $SCO_HOME/.config/profiles/default.toml <<'EOF'
   accept_language = 'zh-CN'

   [default]
   region = 'cn-sh-01'
   region_code = 'cnsh01'

   [regions]
   [regions.cn-sh-01]
   zone = 'cn-sh-01a'
   EOF
   ```

3. 准备任务容器能读到的运行环境。容器只挂载 AFS（`/workspace`），HOME 也不是本账号，所以：
   - `runtime.python`、`tracking.python` 写 `/workspace` 下 conda 环境的绝对路径。
   - W&B 凭据文件放在 `/workspace` 下（权限 600），不要依赖 `~/.bashrc`。
   - `storage.executable` 指向用 `/workspace` 上的 Python 建的 AmphionBucket venv。任务里 `ab` 从 AmphionKeys 加载的 `AMPHION_BUCKET_SENSECORE_*` 读取凭据。

## 配置

| 字段 | 含义 |
| --- | --- |
| `sco_home` | `sco` 安装目录（`SCO_HOME`） |
| `credentials` | AmphionKeys 的 `sensecore.env`，提交和任务内都会加载 |
| `workspace` / `aec2` | 工作空间 `workspace-whai`，集群 `cluster-whai` |
| `image` | 任务镜像 |
| `worker_spec` / `nodes` | 每个节点的规格和节点数；整机 8 卡为 `N3lS.Ii.I60.94c944g` |
| `priority` / `quota_type` / `wait` | 默认 `NORMAL`、`reserved`、`true`（资源不足时排队） |
| `mounts` | `卷ID[/子目录]:挂载路径`；挂载路径须与本机路径一致 |
| `environment` | 传给任务的非敏感环境变量（如 NCCL IB 参数），凭据不能写在这里 |

`runtime.distributed.processes` 必须等于单节点 GPU 数。节点数、rank、master 地址和端口在任务启动时取自平台注入的 `SENSECORE_PYTORCH_NNODES`、`SENSECORE_PYTORCH_NODE_RANK`、`MASTER_ADDR`、`MASTER_PORT`，和配置不一致时直接失败。

## 运行与查看

```bash
open-audio-llm train --config examples/configs/train/sft-acp.yaml --dry-run   # 查看 sco 提交命令，不提交
open-audio-llm train --config examples/configs/train/sft-acp.yaml             # 提交，执行状态为 submitted
```

提交前会检查配置引用的所有本地文件都在挂载路径内，例如 `~/...` 会被拒绝。提交成功后，执行记录的 `status.json` 写入 `cluster.job`（如 `pt-xxxxxxxx`）。任务启动后，rank 0 运行 `open-audio-llm cluster run --attempt <执行目录>`，按普通执行流程更新状态（原提交记录保留在 `history`）；其他节点等 rank 0 进入 `running` 后加入同一个 torchrun。实验编排遇到已提交的任务会停下，等它完成后再运行依赖它的任务。

查看任务：

```bash
source third_party/amphionkeys/load.sh sensecore
export SCO_ACCESS_KEY_ID=$AMPHION_BUCKET_SENSECORE_ACCESS_KEY SCO_ACCESS_KEY_SECRET=$AMPHION_BUCKET_SENSECORE_SECRET_KEY
export SCO_CONFIG=$SCO_HOME/.config SCO_DATA_HOME=$SCO_HOME/.data
sco aec2 clusters usage --name=cluster-whai
sco acp jobs list --workspace-name=workspace-whai
sco acp jobs stream-logs --workspace-name=workspace-whai <job>
sco acp jobs stop --workspace-name=workspace-whai <job>
```

注意：任务如果在 rank 0 启动器运行之前就失败（拉镜像失败、挂载失败等），`status.json` 会停留在 `submitted`，需要用 `sco acp jobs describe` 查看原因。

## 单节点开发任务（dev runner）

清洗、打标、建数据这类不走 `open-audio-llm train` 的任务，可以先提交一个单节点任务，让它在节点上常驻 `scripts/acp_dev_runner.py`，再往目录里放 shell 脚本让它执行。2026-11-08 西语数据（清洗、标点老师、热词、场景池）用的就是这种方式。

1. 把 runner 复制到 `/workspace` 下的固定位置（任务不应依赖工作树当前切在哪个分支），建好目录：

   ```bash
   cp scripts/acp_dev_runner.py /workspace/workspace/<用户>/.local-tools/asr-holder/dev_runner.py
   D=/workspace/workspace/<用户>/scratch/acp-<名字>
   mkdir -p $D/{inbox,running,logs,done} && chmod -R 777 $D   # 任务里以 root 运行
   ```

2. 按上面“查看任务”一节设好 `sco` 环境变量，提交（镜像、规格、挂载同训练配置的 `cluster` 段）：

   ```bash
   sco acp jobs create --workspace-name=workspace-whai --aec2-name=cluster-whai \
     --job-name=AmphionASR-<名字>-runner-1node \
     --container-image-url=registry.cn-sh-01.sensecore.cn/ccr-wh/amphion-base:amphion-base-20260923072955 \
     --training-framework=pytorch --worker-spec=N3lS.Ii.I60.94c944g --worker-nodes=1 \
     --priority=NORMAL --quota-type=reserved \
     --storage-mount=01a0c847-3d8d-7b42-8c75-50cbd189bb56:/workspace \
     "--command=/workspace/workspace/<用户>/miniconda3/envs/oal-train/bin/python /workspace/workspace/<用户>/.local-tools/asr-holder/dev_runner.py $D"
   ```

3. 用法：把 `<名字>.sh` 放进 `$D/inbox/`，runner 5 秒内移到 `running/` 并开始执行，输出在 `logs/<名字>.log`，结束后写 `done/<名字>.exit`。多个脚本同时运行，各自用 `CUDA_VISIBLE_DEVICES` 分卡。
   查看节点状态也一样：放一个只跑 `nvidia-smi` 的脚本，读它的日志。
4. 用完 `sco acp jobs stop --workspace-name=workspace-whai <job>` 释放节点。

注意：

- **不要修改正在运行的脚本**。bash 边执行边读脚本文件，改动会让它读到错位的内容（实际出现过 `unexpected EOF`）。要改就复制一份新名字放进 inbox。
- **停止某个脚本**：没有远程 kill，放一个 `pkill -f <进程特征>` 的脚本进 inbox。runner 本身只能通过停掉 ACP 任务来结束。
- 脚本里在后台常驻服务（如 vLLM）时，**不要用不带参数的 `wait`**，它会连服务一起等，脚本永远不结束。只等具体的进程号，或者给每一轮加 `timeout`。
- 任务里以 root 运行，产物默认归 root。脚本开头写 `umask 000`，本机账号才能继续改写这些文件。
- 本机等待节点上的结果时，同时监控“日志多久没更新”，不要只等结束标记；否则脚本卡住时不会有任何提示。
