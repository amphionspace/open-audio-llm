#!/usr/bin/env bash
# Twice the budget for both designs: 5000 steps each (480,000 samples), to see whether EAGLE-3 catches up.
T=/workspace/workspace/mingdong/repo/oal-mtp/runs/mtp-draft-head-20261031/scripts/train_draft_acp.sh
A=/workspace/workspace/mingdong/repo/oal-mtp/runs/mtp-draft-head-20261031/tasks/01-validate-draft/attempts
bash $T $A/006/artifacts/training 5000 1000 16 eagle1 mtp-draft-head-20261031-train-eagle1-5000; echo "eagle1-5000 exit $?"
bash $T $A/007/artifacts/training 5000 1000 16 eagle3 mtp-draft-head-20261031-train-eagle3-5000; echo "eagle3-5000 exit $?"
