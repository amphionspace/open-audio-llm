#!/usr/bin/env bash
# 03-eval analysis on the dev machine, once acp-node.sh is done: run completeness, then avg3 against the start K on the
# context-correction eval v2 (think off and on), both measured on the same node and code. Outputs: tasks/04-gate/.
# The delivery suites are compared in runs/zh-ctx-grpo-20261120: compare_delivery.py r1117 kgrpo; scene_asr_drops.py r1117 k kgrpo.
# usage: bash scripts/compare.sh
set -u
E=/workspace/workspace/mingdong/repo/oal-ctx-grpo/runs/zh-ctx-grpo-k-20261122; C=$E/tasks/04-gate; mkdir -p $C  # tasks/03-eval was created as root on the node
PY=/workspace/workspace/mingdong/.local-tools/amphion-eval-0.14.0-eedcf15/bin/python
CMP_SRC=/workspace/workspace/mingdong/repo/amphion-eval-ctx-compare/src  # AmphionEval !158 (beb1f99), not in 0.14/0.15
r() { ls -td /workspace/data/ae-runs/ctx-grpo-k-20261122-$1/runs/run-* | head -n 1; }
rc=0
for d in kgrpo-avg3 kgrpo-avg3-think start-k start-k-think; do
  python3 - "$(r $d)" "$d" <<'PYEOF' || rc=1
import json, sys
run, name = sys.argv[1], sys.argv[2]
c = json.load(open(f"{run}/coverage.json")); j = json.load(open(f"{run}/run.json"))
print(name, run.rsplit("/", 1)[-1], c["successful"], "/", c["total"], "complete" if c["complete"] else "INCOMPLETE",
      j.get("wandb_url", ""))
sys.exit(0 if c["complete"] and c["successful"] == c["total"] else 1)
PYEOF
done
[ $rc = 0 ] || { echo "eval runs incomplete"; exit 1; }
for m in "" -think; do
  PYTHONPATH=$CMP_SRC $PY -m amphion_eval.cli open-audio-llm compare "$(r start-k$m)" "$(r kgrpo-avg3$m)" \
    --output $C/cmp-start-k-avg3${m:-"-off"}.json >/dev/null 2>$C/cmp-start-k-avg3${m:-"-off"}.log &
done
wait; ls $C
# 拆分诊断 (README): the hybrid kgrpo-llm against the start K, once acp-swap.sh is done. start-k is this round's run
# (same code-v3 and client, measured on the K GRPO node).
if [ -d /workspace/data/ae-runs/ctx-grpo-k-20261122-kgrpo-llm ]; then
  for m in "" -think; do
    PYTHONPATH=$CMP_SRC $PY -m amphion_eval.cli open-audio-llm compare "$(r start-k$m)" "$(r kgrpo-llm$m)" \
      --output $C/cmp-start-k-kgrpo-llm${m:-"-off"}.json >/dev/null 2>$C/cmp-start-k-kgrpo-llm${m:-"-off"}.log &
    # avg3 -> kgrpo-llm: the same LLM, the audio tower put back to 11.17's (attribution of the context-correction items)
    PYTHONPATH=$CMP_SRC $PY -m amphion_eval.cli open-audio-llm compare "$(r kgrpo-avg3$m)" "$(r kgrpo-llm$m)" \
      --output $C/cmp-avg3-kgrpo-llm${m:-"-off"}.json >/dev/null 2>$C/cmp-avg3-kgrpo-llm${m:-"-off"}.log &
  done
  wait; ls $C
fi
