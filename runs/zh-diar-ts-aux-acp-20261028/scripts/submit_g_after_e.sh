#!/usr/bin/env bash
# E 组训完、节点空出后提交 G 组（F 组缩短到 6000 步），再起 G 组选模。
# 两个节点现由 E、F 占用；等 E 组作业结束后提交。
# 只看 E 组作业的状态，不数作业个数（submit_e_after_d.sh 曾在 D 刚提交、尚未出现在列表里时数到 1 个而同时提交了 E）。
# 用法：submit_g_after_e.sh <E 组作业名>
set -u
C_JOB=$1  # the job whose end frees the node (arm E)
RUN=/workspace/workspace/mingdong/repo/oal-diar-restart/runs/zh-diar-ts-aux-acp-20261028
PY=/workspace/workspace/mingdong/miniconda3/envs/oal-train/bin/python
OAL=/workspace/workspace/mingdong/miniconda3/envs/oal-train/bin/open-audio-llm
LS=/tmp/claude-1001/-workspace-workspace-mingdong-repo-open-audio-llm/8f3e3753-8a5d-4e33-9651-f9cd182783ba/scratchpad/acp_ls_all.py
cd $RUN
while true; do
  state=$(PYTHONPATH=/workspace/workspace/mingdong/repo/open-audio-llm/src timeout 200 $PY $LS | awk -v j=$C_JOB '$1==j{print $NF}')
  case "$state" in RUNNING|STARTING|QUEUEING|"") sleep 60; continue;; esac
  echo "$C_JOB: $state $(date -u +%H:%MZ)"; break
done
[ -e tasks/02-train-g/attempts/001 ] && { echo "train-g attempt 001 exists; NOT submitting"; exit 1; }
name=$(timeout 300 env -u PYTHONPATH $OAL train --config configs/train-g.yaml --dry-run 2>&1 | grep -o -E -- "--job-name=[^ ,\"']+" | head -1)
case "$name" in *-001) ;; *) echo "dry-run job name '$name' does not end with -001; NOT submitting"; exit 1;; esac
timeout 600 env -u PYTHONPATH $OAL train --config configs/train-g.yaml > tracking/submit-train-g.log 2>&1
echo "submit train-g exit=$? $(date -u +%H:%MZ): $(tail -1 tracking/submit-train-g.log)"
[ -e tasks/02-train-g/attempts/001/status.json ] || { echo "no attempt record; selector NOT started"; exit 1; }
setsid nohup env -u PYTHONPATH $OAL prepare --config configs/select-g.yaml < /dev/null > tracking/select-g.out 2>&1 &
echo "select-g started pid=$! $(date -u +%H:%MZ)"
