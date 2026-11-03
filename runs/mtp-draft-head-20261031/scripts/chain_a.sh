#!/usr/bin/env bash
# 009 plain-ASR load, then 010 / 011 TS streaming replay, one attempt after another on GPUs 4-7.
RUN=/workspace/workspace/mingdong/repo/oal-mtp/runs/mtp-draft-head-20261031/scripts/run_lanes.sh
A=/workspace/workspace/mingdong/repo/oal-mtp/runs/mtp-draft-head-20261031/tasks/02-serve-bench/attempts
bash $RUN $A/009 spec configs/spec-check-asr-load-e1e3.yaml 0 1 2 3; echo "009 exit $?"
bash $RUN $A/010 stream configs/stream-e1e3-lane0.yaml configs/stream-e1e3-lane1.yaml configs/stream-e1e3-lane2.yaml configs/stream-e1e3-lane3.yaml; echo "010 exit $?"
bash $RUN $A/011 stream configs/stream-e1e3-nosync-lane0.yaml configs/stream-e1e3-nosync-lane1.yaml configs/stream-e1e3-nosync-lane2.yaml configs/stream-e1e3-nosync-lane3.yaml; echo "011 exit $?"
