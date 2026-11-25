#!/usr/bin/env bash
# 合成数据拆分诊断 (README) on the dev machine, once acp-datasplit.sh is done: context-correction v2 (think off) of
# pm-train, pm-train-ts and pairmerge-g against 11.17, all four measured on the same node and code; also K (start-k,
# this experiment's 11-22 run, same configs from make_eval_configs.py) against 11.17. 不含合成数据的差值合并 adds pm-real
# (acp-pmreal.sh) against 11.17, and G against the control at 11.06 (record only). 续训本身的纠偏收益 adds 10.28 and
# 11.06 (acp-base-ctx.sh): the 11.06-based arms against 11.06; template, H and J are the 11-21 runs of
# runs/zh-ctx-clue-target-20261117 (same eval set; G measured both days agrees, cmp-g-old-g1106). The clean pair
# (runs/zh-ctx-clean-pair-20261125) adds pm-clean against 11.17, and noclean against the control and 11.06. Outputs:
# tasks/09-datasplit/cmp-<a>-<b>.json, skipped when present. The delivery suites are compared in runs/zh-ctx-grpo-20261120:
# compare_delivery.py r1117 pm-train (and pm-train-ts).
# usage: bash scripts/compare_datasplit.sh
set -u
E=/workspace/workspace/mingdong/repo/oal-ctx-grpo/runs/zh-ctx-grpo-k-20261122; C=$E/tasks/09-datasplit; mkdir -p $C
PY=/workspace/workspace/mingdong/.local-tools/amphion-eval-0.14.0-eedcf15/bin/python
CMP_SRC=/workspace/workspace/mingdong/repo/amphion-eval-ctx-compare/src  # AmphionEval !158 (beb1f99), not in 0.14/0.15
declare -A OLD=([template]=template [h]=template-no-synth-ts [j]=template-synth-ts-twins)
r() { local d=/workspace/data/ae-runs/ctx-grpo-k-20261122-$1; [ -n "${OLD[$1]:-}" ] && d=/workspace/data/ae-runs/ctx-clue-target-20261117-${OLD[$1]}
      ls -td $d/runs/run-* | head -n 1; }
for p in r1117:pm-train r1117:pm-train-ts r1117:pairmerge-g r1117:start-k r1117:pm-real control1106:g1106 \
         base1106:base1028 base1106:control1106 base1106:g1106 base1106:h base1106:template base1106:j \
         r1117:pm-clean control1106:noclean1106 base1106:noclean1106; do
  a=${p%:*}; b=${p#*:}
  [ -f $C/cmp-$a-$b.json ] || PYTHONPATH=$CMP_SRC $PY -m amphion_eval.cli open-audio-llm compare "$(r $a)" "$(r $b)" \
    --output $C/cmp-$a-$b.json >/dev/null 2>$C/cmp-$a-$b.log &
done
wait
python3 - $C <<'PYEOF'
import json, sys
from pathlib import Path
for f in sorted(Path(sys.argv[1]).glob("cmp-*.json")):
    cc = json.loads(f.read_text())["context_correction"]
    rows = [("Emilia accuracy", cc["by"]["source"]["mined:emilia_zh"]["mined_accuracy"])]
    rows += [(f"{k} mined accuracy", v["mined_accuracy"]) for k, v in cc["by"]["train_overlap"].items()]
    rows += [(k, cc["overall"][k]) for k in ("mined_accuracy", "keep_rate", "over_correction_rate", "cer")]
    print(f.stem, "; ".join(f"{k} {100 * x['a']:.2f} -> {100 * x['b']:.2f}{'*' if x['significant'] else ''}"
                            f" [{100 * x['ci'][0]:+.2f}, {100 * x['ci'][1]:+.2f}]" for k, x in rows))
PYEOF
