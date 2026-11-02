#!/usr/bin/env bash
# Third budget point: 10000 steps each (960,000 samples), about 77 minutes per design.
T=/workspace/workspace/mingdong/repo/oal-mtp/runs/mtp-draft-head-20261031/scripts/train_draft_acp.sh
A=/workspace/workspace/mingdong/repo/oal-mtp/runs/mtp-draft-head-20261031/tasks/01-validate-draft/attempts
bash $T $A/010/artifacts/training 10000 2000 16 eagle1 mtp-draft-head-20261031-train-eagle1-10000; echo "eagle1-10000 exit $?"
bash $T $A/011/artifacts/training 10000 2000 16 eagle3 mtp-draft-head-20261031-train-eagle3-10000; echo "eagle3-10000 exit $?"
