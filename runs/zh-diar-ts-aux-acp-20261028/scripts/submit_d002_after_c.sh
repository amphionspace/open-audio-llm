#!/usr/bin/env bash
# D 组（时间刻度）attempt 001 在 248 步因 MaxLengthError 失败（max_length 已调到 16896）。两个节点现由 C、E 占用，
# 等 C 组作业结束、节点空出后提交 D 组 attempt 002，再起 D 组选模（select-d.yaml 已指向 attempts/002）。
# 只看 C 组作业的状态，不数作业个数（submit_e_after_d.sh 曾在 D 刚提交、尚未出现在列表里时数到 1 个而同时提交了 E）。
# 用法：submit_d002_after_c.sh <C 组作业名>
set -u
C_JOB=$1
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
[ -e tasks/02-train-d/attempts/002 ] && { echo "train-d attempt 002 exists; NOT submitting"; exit 1; }
name=$(timeout 300 env -u PYTHONPATH $OAL train --config configs/train-d.yaml --dry-run 2>&1 | grep -o -E -- "--job-name=[^ ,\"']+" | head -1)
case "$name" in *-002) ;; *) echo "dry-run job name '$name' does not end with -002; NOT submitting"; exit 1;; esac
timeout 600 env -u PYTHONPATH $OAL train --config configs/train-d.yaml > tracking/submit-train-d-002.log 2>&1
echo "submit train-d 002 exit=$? $(date -u +%H:%MZ): $(tail -1 tracking/submit-train-d-002.log)"
[ -e tasks/02-train-d/attempts/002/status.json ] || { echo "no attempt record; selector NOT started"; exit 1; }
setsid nohup env -u PYTHONPATH $OAL prepare --config configs/select-d.yaml < /dev/null > tracking/select-d-002.out 2>&1 &
echo "select-d started pid=$! $(date -u +%H:%MZ)"
