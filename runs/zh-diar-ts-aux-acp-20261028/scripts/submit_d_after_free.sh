#!/usr/bin/env bash
# A 组或 C 组的 ACP 作业结束（选模的停止请求或训满）、节点空出后，提交 D 组（时间刻度），再在开发机上起 D 组选模。
# 用法：submit_d_after_free.sh <A 组作业名> <C 组作业名>
# 每 15 秒看一次两个作业，任一离开 RUNNING / STARTING / QUEUEING 后 → A 组预检已通过且数据配置与 A 组实际训练的相同（D 组用同一份
# train-data-a.yaml）、attempts/001 不存在、dry-run 任务名以 -001 结尾 → 提交（不带 PYTHONPATH）。任一条件不满足只记录、不提交。
set -u
A_JOB=$1 C_JOB=$2
RUN=/workspace/workspace/mingdong/repo/oal-diar-restart/runs/zh-diar-ts-aux-acp-20261028
PY=/workspace/workspace/mingdong/miniconda3/envs/oal-train/bin/python
OAL=/workspace/workspace/mingdong/miniconda3/envs/oal-train/bin/open-audio-llm
LS=/tmp/claude-1001/-workspace-workspace-mingdong-repo-open-audio-llm/8f3e3753-8a5d-4e33-9651-f9cd182783ba/scratchpad/acp_ls_all.py
cd $RUN
while true; do
  states=$(PYTHONPATH=/workspace/workspace/mingdong/repo/open-audio-llm/src timeout 200 $PY $LS |
    awk -v a=$A_JOB -v c=$C_JOB '$1==a||$1==c{print $1, $NF}')
  ended=$(echo "$states" | awk 'NF==2 && $2!="RUNNING" && $2!="STARTING" && $2!="QUEUEING"{print}')
  # Both jobs must be listed: an empty or partial listing (sco timeout) is not an ended job.
  [ "$(echo "$states" | awk 'NF==2' | wc -l)" = 2 ] && [ -n "$ended" ] && { echo "$ended $(date -u +%H:%MZ)"; break; }
  sleep 15
done
grep -q "Catalog data preflight passed" tracking/preflight-a.out || { echo "preflight a not passed; NOT submitting"; exit 1; }
# make_recipes.py rewrites every arm's data config, so compare content with what arm A trains on, not mtimes.
$PY -c "import sys, yaml; sys.exit(yaml.safe_load(open('configs/train-data-a.yaml')) != yaml.safe_load(open('tasks/02-train-a/attempts/001/data-effective.yaml')))" ||
  { echo "train-data-a.yaml differs from arm A's data-effective.yaml; NOT submitting"; exit 1; }
[ -e tasks/02-train-d/attempts/001 ] && { echo "train-d attempt 001 exists; NOT submitting"; exit 1; }
name=$(timeout 300 env -u PYTHONPATH $OAL train --config configs/train-d.yaml --dry-run 2>&1 | grep -o -E -- "--job-name=[^ ,\"']+" | head -1)
case "$name" in *-001) ;; *) echo "dry-run job name '$name' does not end with -001; NOT submitting"; exit 1;; esac
timeout 600 env -u PYTHONPATH $OAL train --config configs/train-d.yaml > tracking/submit-train-d.log 2>&1
echo "submit train-d exit=$? $(date -u +%H:%MZ): $(tail -1 tracking/submit-train-d.log)"
[ -e tasks/02-train-d/attempts/001/status.json ] || { echo "no attempt record; selector NOT started"; exit 1; }
setsid nohup env -u PYTHONPATH $OAL prepare --config configs/select-d.yaml < /dev/null > tracking/select-d.out 2>&1 &
echo "select-d started pid=$! $(date -u +%H:%MZ)"
