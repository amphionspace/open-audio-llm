"""What a plain-ASR drift is made of, on one source, against a base model.

- S / D / I totals per model (AmphionEval tokens and normalization);
- the character edits that grew most (base -> model): deleted char, inserted char, substitution ref->hyp;
- per-utterance error deltas: how many utterances got worse / better, and whether the same utterances get worse in
  every model (a shared, systematic cause) or in different ones (decoding noise around a small shift);
- delta by audio duration bin (audio_slots[0].ref.duration).
Usage (amphion-eval python): asr_drift_ops.py SOURCE_PREFIX SAMPLES.jsonl BASE_RUN NAME=RUN [NAME=RUN ...]
The last NAME is the one whose edits are listed."""
import collections
import json
import sys

from amphion_eval.integrations.open_audio_llm import scoring


def align(ref, hyp):
    """Levenshtein alignment; returns [(op, ref_char, hyp_char)] with op in S/D/I (matches dropped)."""
    n, m = len(ref), len(hyp)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]))
    ops, i, j = [], n, m
    while i or j:
        if i and j and d[i][j] == d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            if ref[i - 1] != hyp[j - 1]:
                ops.append(("S", ref[i - 1], hyp[j - 1]))
            i, j = i - 1, j - 1
        elif i and d[i][j] == d[i - 1][j] + 1:
            ops.append(("D", ref[i - 1], "")); i -= 1
        else:
            ops.append(("I", "", hyp[j - 1])); j -= 1
    return ops


def load(run, prefix, lang):
    out = {}
    with open(f"{run}/predictions.jsonl") as f:
        for line in f:
            p = json.loads(line)
            if p["source"].startswith(prefix):
                ref = scoring.tokens(p.get("reference") or "", lang)
                hyp = scoring.tokens(p.get("prediction") or "", lang)
                out[p["id"]] = (ref, hyp, align(ref, hyp) if len(ref) * max(len(hyp), 1) < 250000 else None)
    return out


def main():
    prefix, samples_path, base_run = sys.argv[1], sys.argv[2], sys.argv[3]
    models = [a.split("=", 1) for a in sys.argv[4:]]
    dur = {}
    with open(samples_path) as f:
        for line in f:
            s = json.loads(line)
            if s["source"].startswith(prefix):
                dur[s["id"]] = s["record"]["audio_slots"][0]["ref"].get("duration")
                lang = scoring.variety(s["language"])  # the summary's token kind ("zh" / "en")
    base = load(base_run, prefix, lang)
    outs = {name: load(run, prefix, lang) for name, run in models}
    ids = sorted(i for i in base if all(i in o for o in outs.values()) and base[i][2] is not None
                 and all(o[i][2] is not None for o in outs.values()))
    nref = sum(len(base[i][0]) for i in ids)
    print(f"{prefix}: {len(ids)} utterances, {nref} reference chars (runaway alignments skipped: {len(base) - len(ids)})")
    def totals(o):
        c = collections.Counter(op for i in ids for op, _, _ in o[i][2])
        return c
    b = totals(base)
    print(f"  base  S {b['S']:6d} D {b['D']:6d} I {b['I']:6d}  CER {(b['S'] + b['D'] + b['I']) / nref * 100:.2f}")
    for name, o in outs.items():
        c = totals(o)
        print(f"  {name:5s} S {c['S']:6d} ({c['S'] - b['S']:+5d}) D {c['D']:6d} ({c['D'] - b['D']:+5d}) I {c['I']:6d} ({c['I'] - b['I']:+5d})"
              f"  CER {(c['S'] + c['D'] + c['I']) / nref * 100:.2f}")

    err = lambda o, i: len(o[i][2])
    names = list(outs)
    print("\nper-utterance deltas vs base")
    for name in names:
        o = outs[name]; dl = [err(o, i) - err(base, i) for i in ids]
        worse = sum(x > 0 for x in dl); better = sum(x < 0 for x in dl)
        print(f"  {name:5s} worse {worse:5d} (+{sum(x for x in dl if x > 0)})  better {better:5d} ({sum(x for x in dl if x < 0)})  net {sum(dl):+d}")
    if len(names) > 1:
        last = names[-1]
        worse_last = [i for i in ids if err(outs[last], i) > err(base, i)]
        shared = [i for i in worse_last if all(err(outs[n], i) > err(base, i) for n in names[:-1])]
        net_shared = sum(err(outs[last], i) - err(base, i) for i in shared)
        print(f"  {last} worse utterances also worse in every other model: {len(shared)} of {len(worse_last)}, "
              f"their {last} delta +{net_shared}")
        # Paired: total delta of the shared-worse set in the other models (does it reproduce?)
        for n in names[:-1]:
            print(f"    same set in {n}: {sum(err(outs[n], i) - err(base, i) for i in shared):+d}")
        # Mean pairwise delta correlation
        import statistics
        for n in names[:-1]:
            x = [err(outs[n], i) - err(base, i) for i in ids]; y = [err(outs[last], i) - err(base, i) for i in ids]
            print(f"    corr(delta {n}, delta {last}) {statistics.correlation(x, y):+.3f}")
        # Self-noise reference: two models' disagreement relative to each other vs to base
        for n in names[:-1]:
            dd = sum(abs(err(outs[n], i) - err(outs[last], i)) for i in ids)
            db = sum(abs(err(outs[last], i) - err(base, i)) for i in ids)
            print(f"    sum|err {n} - err {last}| {dd}   sum|err {last} - err base| {db}")

    last = names[-1]; o = outs[last]
    print(f"\n{last}: character edits that grew most vs base (count base -> {last}, delta)")
    def ops(src):
        c = collections.Counter()
        for i in ids:
            for op, r, h in src[i][2]:
                c[(op, r, h)] += 1
        return c
    cb, cm = ops(base), ops(o)
    keys = set(cb) | set(cm)
    grow = sorted(keys, key=lambda k: cm[k] - cb[k], reverse=True)
    for kind in ("D", "I", "S"):
        rows = [k for k in grow if k[0] == kind][:15]
        tot = sum(cm[k] - cb[k] for k in keys if k[0] == kind)
        print(f"  {kind} (net {tot:+d}): " + "  ".join(f"{k[1] or '∅'}→{k[2] or '∅'} {cb[k]}→{cm[k]}" for k in rows))
    # How concentrated are the grown edits? share of net growth from the top 20 edit types
    pos = sorted((cm[k] - cb[k] for k in keys), reverse=True)
    net = sum(pos)
    print(f"  net edit growth {net:+d}; top-20 edit types sum {sum(pos[:20]):+d}; edit types that grew {sum(x > 0 for x in pos)}, shrank {sum(x < 0 for x in pos)}")

    print(f"\n{last}: delta by duration (s)")
    bins = (0, 2, 4, 6, 10, 15, 1000)
    for lo, hi in zip(bins, bins[1:]):
        sel = [i for i in ids if dur.get(i) is not None and lo <= dur[i] < hi]
        if sel:
            r = sum(len(base[i][0]) for i in sel)
            print(f"  [{lo:>3}, {hi:>4})  n {len(sel):5d}  ref {r:6d}  base {sum(err(base, i) for i in sel) / r * 100:5.2f}  "
                  + "  ".join(f"{n} {sum(err(outs[n], i) for i in sel) / r * 100:5.2f}" for n in names))


main()
