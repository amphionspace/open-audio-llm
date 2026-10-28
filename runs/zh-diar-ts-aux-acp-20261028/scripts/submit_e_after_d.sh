#!/usr/bin/env bash
# 第二个节点空出后提交 E 组（Emilia2 换 MOSS Pro 第二批重标），再在开发机上起 E 组选模。第一个空出的节点归 D 组（submit_d_after_free.sh）。
# 每 60 秒看一次：D 组 attempts/001 已存在，且本轮（AmphionASR-zh-diar-restart-*）处于 RUNNING / STARTING / QUEUEING 的作业少于 2 个
# （用户 2026-10-28：本项目可用两个节点）→ E 组预检已通过、attempts/001 不存在、dry-run 任务名以 -001 结尾 → 提交（不带 PYTHONPATH）。
# 任一条件不满足只记录、不提交。作业列表取不到（sco 超时）时不算节点空出。
set -u
RUN=/workspace/workspace/mingdong/repo/oal-diar-restart/runs/zh-diar-ts-aux-acp-20261028
PY=/workspace/workspace/mingdong/miniconda3/envs/oal-train/bin/python
OAL=/workspace/workspace/mingdong/miniconda3/envs/oal-train/bin/open-audio-llm
LS=/tmp/claude-1001/-workspace-workspace-mingdong-repo-open-audio-llm/8f3e3753-8a5d-4e33-9651-f9cd182783ba/scratchpad/acp_ls_all.py
cd $RUN
while true; do
  if [ -e tasks/02-train-d/attempts/001 ]; then
    listing=$(PYTHONPATH=/workspace/workspace/mingdong/repo/open-audio-llm/src timeout 200 $PY $LS)
    if [ -n "$listing" ]; then
      active=$(echo "$listing" | awk '$2 ~ /^AmphionASR-zh-diar-restart-/ && ($NF=="RUNNING" || $NF=="STARTING" || $NF=="QUEUEING")' | wc -l)
      [ "$active" -lt 2 ] && { echo "active diarization jobs: $active $(date -u +%H:%MZ)"; break; }
    fi
  fi
  sleep 60
done
grep -q "Catalog data preflight passed" tracking/preflight-e.out || { echo "preflight e not passed; NOT submitting"; exit 1; }
[ -e tasks/02-train-e/attempts/001 ] && { echo "train-e attempt 001 exists; NOT submitting"; exit 1; }
name=$(timeout 300 env -u PYTHONPATH $OAL train --config configs/train-e.yaml --dry-run 2>&1 | grep -o -E -- "--job-name=[^ ,\"']+" | head -1)
case "$name" in *-001) ;; *) echo "dry-run job name '$name' does not end with -001; NOT submitting"; exit 1;; esac
timeout 600 env -u PYTHONPATH $OAL train --config configs/train-e.yaml > tracking/submit-train-e.log 2>&1
echo "submit train-e exit=$? $(date -u +%H:%MZ): $(tail -1 tracking/submit-train-e.log)"
[ -e tasks/02-train-e/attempts/001/status.json ] || { echo "no attempt record; selector NOT started"; exit 1; }
setsid nohup env -u PYTHONPATH $OAL prepare --config configs/select-e.yaml < /dev/null > tracking/select-e.out 2>&1 &
echo "select-e started pid=$! $(date -u +%H:%MZ)"
