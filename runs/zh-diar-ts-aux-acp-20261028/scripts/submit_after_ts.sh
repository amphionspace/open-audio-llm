#!/usr/bin/env bash
# 一个 TS 训练任务（ACP 作业）结束、节点空出后，在它的节点上接本轮的一组（A 或 B），再在开发机上起这一组的选模。
# 用法：submit_after_ts.sh <组 a|b> <TS ACP 作业名>
# 每 15 秒看一次那个 TS 作业，离开 RUNNING / STARTING / QUEUEING 后 → 本组预检已通过、attempts/001 不存在、dry-run 任务名以 -001 结尾
# → 提交（不带 PYTHONPATH，避免用到别的代码快照）。A 组提交后直接起选模（测 step 0 基线）；B 组等 A 组的基线写出后再起选模。
# 任一条件不满足只记录、不提交。
set -u
ARM=$1 TS_JOB=$2
RUN=/workspace/workspace/mingdong/repo/oal-diar-restart/runs/zh-diar-ts-aux-acp-20261028
PY=/workspace/workspace/mingdong/miniconda3/envs/oal-train/bin/python
OAL=/workspace/workspace/mingdong/miniconda3/envs/oal-train/bin/open-audio-llm
LS=/tmp/claude-1001/-workspace-workspace-mingdong-repo-open-audio-llm/8f3e3753-8a5d-4e33-9651-f9cd182783ba/scratchpad/acp_ls_all.py
cd $RUN
while true; do
  state=$(PYTHONPATH=/workspace/workspace/mingdong/repo/open-audio-llm/src timeout 200 $PY $LS | awk -v j=$TS_JOB '$1==j{print $NF}')
  case "$state" in RUNNING|STARTING|QUEUEING|"") sleep 15; continue;; esac
  echo "$TS_JOB: $state $(date -u +%H:%MZ)"; break
done
grep -q "Catalog data preflight passed" tracking/preflight-$ARM.out || { echo "preflight $ARM not passed; NOT submitting"; exit 1; }
[ -e tasks/02-train-$ARM/attempts/001 ] && { echo "train-$ARM attempt 001 exists; NOT submitting"; exit 1; }
name=$(timeout 300 env -u PYTHONPATH $OAL train --config configs/train-$ARM.yaml --dry-run 2>&1 | grep -o -E -- "--job-name=[^ ,\"']+" | head -1)
case "$name" in *-001) ;; *) echo "dry-run job name '$name' does not end with -001; NOT submitting"; exit 1;; esac
timeout 600 env -u PYTHONPATH $OAL train --config configs/train-$ARM.yaml > tracking/submit-train-$ARM.log 2>&1
echo "submit train-$ARM exit=$? $(date -u +%H:%MZ): $(tail -1 tracking/submit-train-$ARM.log)"
[ -e tasks/02-train-$ARM/attempts/001/status.json ] || { echo "no attempt record; selector NOT started"; exit 1; }
if [ "$ARM" = b ]; then
  for i in $(seq 1 360); do [ -f tasks/03-select-a/attempts/001/artifacts/baseline.json ] && break; sleep 60; done
  [ -f tasks/03-select-a/attempts/001/artifacts/baseline.json ] || { echo "arm A baseline missing after 6 h; select-b NOT started"; exit 1; }
fi
setsid nohup env -u PYTHONPATH $OAL prepare --config configs/select-$ARM.yaml < /dev/null > tracking/select-$ARM.out 2>&1 &
echo "select-$ARM started pid=$! $(date -u +%H:%MZ)"
