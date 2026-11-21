#!/usr/bin/env bash
# The whole experiment on one ACP node: 01-mining (scripts/mining.sh) -> 02-grpo (scripts/train.sh) -> 03-eval
# (scripts/eval_after_training.sh). Start arm J = template-synth-ts-twins (zh-ctx-clue-target-20261117 questions 7, 8);
# targets context_correction_clean_zh@grpo-targets-v2-20261121.
# usage: acp-node.sh [stage ...]   (stages: mining grpo eval; default all three, in that order)
set -u; umask 002
E=/workspace/workspace/mingdong/repo/oal-ctx-grpo/runs/zh-ctx-grpo-20261120; cd $E
TARGETS=/workspace/data/datasets/context_correction_clean_zh/versions/grpo-targets-v2-20261121/targets.jsonl
PROXY=http://10.119.17.252:17920
now() { date -u +%FT%TZ; }
for i in $(seq 60); do  # the forward only lets registered node IPs through; mining and training log to W&B
  code=$(curl -s -m 10 -o /dev/null -w '%{http_code}' -x $PROXY https://api.wandb.ai/healthz)
  [ "$code" = 200 ] && break
  echo "W&B through $PROXY: $code, waiting for the node IP to be let through $(now)"; sleep 30
done
[ "$code" = 200 ] || exit 2
STAGES=("$@"); [ ${#STAGES[@]} -gt 0 ] || STAGES=(mining grpo eval)
for stage in "${STAGES[@]}"; do
  echo "$stage $(now)"
  case $stage in
    mining) bash scripts/mining.sh template-synth-ts-twins $TARGETS || exit $?;;
    grpo) bash scripts/train.sh || exit $?;;
    eval) bash scripts/eval_after_training.sh || exit $?;;
    *) echo "unknown stage $stage"; exit 2;;
  esac
done
echo "done $(now)"
