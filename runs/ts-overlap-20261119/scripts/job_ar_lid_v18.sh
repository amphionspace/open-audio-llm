#!/usr/bin/env bash
# tasks/30 language-ID localization: ar-asr-test (v6, pinned venv) on 11.17's pre-Spanish-increment parent
# (amphionasr2-grpo-v18avg3-ckpt400 = 11.17 minus the es116 increment), dev GPUs 4-5.
set -u; umask 000
E=/workspace/workspace/mingdong/repo/oal-clean-teacher/runs/ts-overlap-20261119; D=/workspace/data/ts-overlap-20261119
L=$D/task30-ar-lid/logs; cd $E; unset no_proxy NO_PROXY
export AE_ENV=$D/ar-baseline/ae-ca7d87f-venv TEMPLATES=$E/configs/ar-templates WB_PROXY=http://127.0.0.1:7890
bash scripts/eval_lanes.sh amphionasr2-grpo-v18avg3-ckpt400 arlid-v18avg3 4 19970 "ar/arasr" > $L/eval-v18avg3.log 2>&1
echo "v18avg3 rc=$? $(date -u +%FT%TZ)" >> $L/eval.txt
