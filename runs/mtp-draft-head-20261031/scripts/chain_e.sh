#!/usr/bin/env bash
# Speed protocol on the 10000-step heads. needs four otherwise idle A800s (the configs name GPUs 0-3), about 80 minutes.
RUN=/workspace/workspace/mingdong/repo/oal-mtp/runs/mtp-draft-head-20261031/scripts/run_lanes.sh
A=/workspace/workspace/mingdong/repo/oal-mtp/runs/mtp-draft-head-20261031/tasks/02-serve-bench/attempts
for n in 015 016 017 018; do mkdir -p $A/$n; done
bash $RUN $A/015 spec configs/spec-check-final.yaml 0 1 2 3; echo "015 exit $?"
bash $RUN $A/016 spec configs/spec-check-asr-load-final.yaml 0 1 2 3; echo "016 exit $?"
bash $RUN $A/017 stream configs/stream-final-lane0.yaml configs/stream-final-lane1.yaml configs/stream-final-lane2.yaml configs/stream-final-lane3.yaml; echo "017 exit $?"
bash $RUN $A/018 stream configs/stream-final-nosync-lane0.yaml configs/stream-final-nosync-lane1.yaml configs/stream-final-nosync-lane2.yaml configs/stream-final-nosync-lane3.yaml; echo "018 exit $?"
