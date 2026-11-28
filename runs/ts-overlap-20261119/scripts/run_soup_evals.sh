#!/usr/bin/env bash
# tasks/15 on one ACP node: weight averages of tasks/13 avg3 (presence drifts to rejection: v12 FA 277 / miss 411) and
# tasks/14 avg3 (drifts to acceptance: FA 451 / miss 278), both GRPO from 11.17 with the ts-hard plan, differing only
# in the ts-present share (0.06 / 0.12). soup = their mean (MTP averaged too); soup-interp050 = mean of soup and 11.17.
# Then the four 11.17 suites for both, side by side (as run_grpo_hard.sh).
# usage: run_soup_evals.sh
set -u; umask 000
E=/workspace/workspace/mingdong/repo/oal-clean-teacher/runs/ts-overlap-20261119; cd $E
ENVS=/workspace/workspace/mingdong/miniconda3/envs; T=tasks/15-grpo-soup; mkdir -p $T
M=/workspace/workspace/mingdong/scratch/acp-dev/ae-models.sh
AVG=/workspace/workspace/mingdong/scratch/ts033/avg_ckpt.py; PY=$ENVS/oal-train/bin/python
export WB_PROXY=http://10.119.17.252:17922
START=/workspace/data/ae-state/model-cache/entries/model-afea989da22b77adc0ce9365
now() { date -u +%FT%TZ; }
build() {  # NAME SRC...
  local n=$1 out w; shift
  if out=$(bash $M fetch $n 2>/dev/null); then echo "$out" | tail -1; return 0; fi
  w=$(bash $M stage) || return 4; $PY $AVG $w "$@" >&2 || return 4; rm -f $w/.complete
  bash $M put $w --name $n | tail -1
}
A13=$(bash $M fetch ts-overlap-grpo-hard-001-avg3 | tail -1) || exit 4
A14=$(bash $M fetch ts-overlap-grpo-present-001-avg3 | tail -1) || exit 4
S=$(build ts-overlap-grpo-soup-t13t14-avg3 $A13 $A14) || exit 4; echo "soup $S $(now)"
I=$(build ts-overlap-grpo-soup-t13t14-interp050 $START $S) || exit 4; echo "soup-interp050 $I $(now)"
[ -d $S/mtp ] || { echo "soup has no mtp/"; exit 4; }
rc=0; pids=(); gb=0; pb=19900
for x in avg3 interp050; do
  bash scripts/eval_lanes.sh ts-overlap-grpo-soup-t13t14-$x soup-$x $gb $pb \
    "eval-sumfull-es116-core/v12 eval-sumfull-es116-core/tsen" \
    "eval-v18avg3-ckpt400-asrfull/asrfull eval-sumfull-es116-core/custts" > $T/eval-$x.log 2>&1 & pids+=($!)
  gb=$((gb + 4)); pb=$((pb + 100)); sleep 60
done
for x in 0 1; do wait ${pids[$x]} || rc=1; done
echo "evals done rc=$rc $(now)"; exit $rc
