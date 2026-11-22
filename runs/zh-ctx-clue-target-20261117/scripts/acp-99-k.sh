#!/usr/bin/env bash
# zh-ctx-clue-target-20261117, question 9 (K: J's recipe on the final texts, plus plain-ASR TTS dialogues and es116's
# Spanish, trained from 11.17), on one ACP node: train template-twins-dialog-final-1117 on GPUs 0-7, fetch its
# checkpoint-3000, then at the same time its context-correction eval v2 (node_chain.sh eval, GPUs 0/1) and the 27
# delivery suites (GRPO experiment scripts/acp-delivery-k.sh, GPUs 2-5), vLLM starts staggered.
set -u; umask 002
E=/workspace/workspace/mingdong/repo/oal-clue-target/runs/zh-ctx-clue-target-20261117; C=$E/scripts/node_chain.sh
G=/workspace/workspace/mingdong/repo/oal-ctx-grpo/runs/zh-ctx-grpo-20261120
MODELS=/workspace/workspace/mingdong/scratch/acp-dev/ae-models.sh
ARM=template-twins-dialog-final-1117
now() { date -u +%FT%TZ; }
ARMS=$ARM bash $C train || exit $?
cd $E
a=; for d in tasks/01-train-$ARM/attempts/*/; do grep -q '"execution": "completed"' $d/status.json 2>/dev/null && a=$(basename $d); done
[ -n "$a" ] || { echo "no completed training attempt"; exit 5; }
K=$(bash $MODELS fetch "whai:open-audio-llm/runs/zh-ctx-clue-target-20261117/tasks/01-train-$ARM/attempts/$a/artifacts/training/checkpoint-3000/") || exit 5
echo "$ARM checkpoint $K $(now)"
ARMS=$ARM bash $C eval & c=$!
sleep 120  # after the context-correction servers have started
bash $G/scripts/acp-delivery-k.sh "$K" & s=$!
wait $c; rc=$?; wait $s; rs=$?
echo "ctx eval K exit $rc, delivery K exit $rs $(now)"
[ "$rc" = 0 ] && [ "$rs" = 0 ]
