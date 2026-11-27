#!/usr/bin/env bash
# tasks/29: admit each training job's node 0 on this dev machine's W&B forwarder 10.119.18.89:17922 (-> local proxy 7890) and
# raise its wandb-ready flag. One forwarder serves every job: it is restarted with the union of the admitted IPs.
# usage (dev machine, background): admit_t29.sh OUT_DIR [OUT_DIR ...]
set -u
FWD=/workspace/workspace/mingdong/.claude/tmp/claude-1001/-workspace-workspace-mingdong-repo-open-audio-llm/318bdda7-689c-4e40-984e-5390ca3a2d12/scratchpad/tcp_forward.py
LOG=/workspace/data/ts-overlap-20261119/task29-simneg-asranchor/logs/forward17922.log
ips=(); left=("$@")
while [ ${#left[@]} -gt 0 ]; do
  rest=()
  for O in "${left[@]}"; do
    if [ -s $O/logs/node-0.txt ]; then
      ips+=("$(awk '{print $6}' $O/logs/node-0.txt)"); allow=$(IFS=,; echo "${ips[*]}")
      pkill -f "^python3 -I $FWD 17922" ; sleep 2
      setsid nohup python3 -I $FWD 17922 7890 10.119.18.89 $allow >> $LOG 2>&1 < /dev/null &
      sleep 5
      if pgrep -af "tcp_forward.py 17922" | grep -q -- "$allow"; then touch $O/wandb-ready; echo "admitted $allow for $O $(date -u +%FT%TZ)"
      else echo "forwarder not up for $allow"; exit 1; fi
    else rest+=("$O"); fi
  done
  left=("${rest[@]+"${rest[@]}"}"); [ ${#left[@]} -gt 0 ] && sleep 30
done
