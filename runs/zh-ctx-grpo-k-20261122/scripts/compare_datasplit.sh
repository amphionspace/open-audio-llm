#!/usr/bin/env bash
# 合成数据拆分诊断 (README) on the dev machine, once acp-datasplit.sh is done: context-correction v2 (think off) of
# pm-train, pm-train-ts and pairmerge-g against 11.17, all four measured on the same node and code; also K (start-k,
# this experiment's 11-22 run, same configs from make_eval_configs.py) against 11.17. Outputs:
# tasks/09-datasplit/cmp-r1117-<name>.json, skipped when present. The delivery suites are compared in runs/zh-ctx-grpo-20261120:
# compare_delivery.py r1117 pm-train (and pm-train-ts).
# usage: bash scripts/compare_datasplit.sh
set -u
E=/workspace/workspace/mingdong/repo/oal-ctx-grpo/runs/zh-ctx-grpo-k-20261122; C=$E/tasks/09-datasplit; mkdir -p $C
PY=/workspace/workspace/mingdong/.local-tools/amphion-eval-0.14.0-eedcf15/bin/python
CMP_SRC=/workspace/workspace/mingdong/repo/amphion-eval-ctx-compare/src  # AmphionEval !158 (beb1f99), not in 0.14/0.15
r() { ls -td /workspace/data/ae-runs/ctx-grpo-k-20261122-$1/runs/run-* | head -n 1; }
for n in pm-train pm-train-ts pairmerge-g start-k; do
  [ -f $C/cmp-r1117-$n.json ] || PYTHONPATH=$CMP_SRC $PY -m amphion_eval.cli open-audio-llm compare "$(r r1117)" "$(r $n)" \
    --output $C/cmp-r1117-$n.json >/dev/null 2>$C/cmp-r1117-$n.log &
done
wait
python3 - $C <<'PYEOF'
import json, sys
from pathlib import Path
for f in sorted(Path(sys.argv[1]).glob("cmp-r1117-*.json")):
    cc = json.loads(f.read_text())["context_correction"]
    rows = [("Emilia accuracy", cc["by"]["source"]["mined:emilia_zh"]["mined_accuracy"])]
    rows += [(f"{k} mined accuracy", v["mined_accuracy"]) for k, v in cc["by"]["train_overlap"].items()]
    rows += [(k, cc["overall"][k]) for k in ("mined_accuracy", "keep_rate", "over_correction_rate", "cer")]
    print(f.stem, "; ".join(f"{k} {100 * x['a']:.2f} -> {100 * x['b']:.2f}{'*' if x['significant'] else ''}"
                            f" [{100 * x['ci'][0]:+.2f}, {100 * x['ci'][1]:+.2f}]" for k, x in rows))
PYEOF
