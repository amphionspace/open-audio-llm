"""README "风险" 1 (second K0 seed, tasks 05-07): gate.py's gates for two more pairs, written to tasks/08-eval-k0s2/<pair>/.
  k0s2grpo-vs-kgrpo   the gate again against the second K0 (the package's gain and regressions)
  k0grpo-vs-k0s2grpo  two K0 runs, no package on either side: what run-to-run noise alone shows under the same rule
  start-k0-vs-start-k0s2  record only: context correction v2 of the two K0 SFTs (no delivery suites for SFT models)
usage: gate_seed2.py   (after eval_after_training.sh k0s2, compare_delivery.py k0s2grpo kgrpo and k0grpo k0s2grpo)
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gate  # noqa: E402

R = gate.RUNS
ROOT = {"kgrpo": R / "ctx-grpo-k-20261122-kgrpo-avg3", "k0grpo": R / "ctx-mainline-pair-20261127-k0grpo-avg3",
        "k0s2grpo": R / "ctx-mainline-pair-20261127-k0s2grpo-avg3", "start-k0": R / "ctx-mainline-pair-20261127-start-k0",
        "start-k0s2": R / "ctx-mainline-pair-20261127-start-k0s2"}
for base, cand in (("k0s2grpo", "kgrpo"), ("k0grpo", "k0s2grpo")):
    gate.CTX = {"kgrpo": ROOT[cand], "k0grpo": ROOT[base]}  # gate.py's keys are roles: candidate, baseline
    gate.MODELS = (base, cand)
    gate.CMP = gate.G / "tasks/04-delivery/compare" / f"{base}-vs-{cand}"
    gate.OUT = gate.HERE / "tasks/08-eval-k0s2" / f"{base}-vs-{cand}"
    gate.OUT.mkdir(parents=True, exist_ok=True)
    gate.TITLE = (f"# 判定：{cand} 对 {base}，think 关，配对 bootstrap 95% CI，差 = {cand} − {base}（百分点）"
                  + ("" if cand == "kgrpo" else "\n\n两边都不带纠偏这一包：这里的“收益”“回退”只是运行间波动，第 3 节的结论规则不适用。"))
    print(f"== {base} -> {cand}")
    gate.main()
gate.CTX = {"kgrpo": ROOT["start-k0s2"], "k0grpo": ROOT["start-k0"]}
ctx, _ = gate.ctx_gate()
out = gate.HERE / "tasks/08-eval-k0s2/start-k0-vs-start-k0s2"
out.mkdir(parents=True, exist_ok=True)
(out / "ctx.json").write_text(json.dumps(ctx, ensure_ascii=False, indent=1))
print("== start-k0 -> start-k0s2 (ctx v2, think off)")
for x in ctx:
    print(gate.fmt(x, f"{x['source']} · {x['group']} · {x['metric']}（{x['items']}）"))
