"""Where a model's extra v12 false alarms (target absent, text output) come from, against a base model, per utterance.

- per scenario family (scenario label with -neg / -enrollmix / k-count suffixes kept: v12 scenarios are already the
  design cells): FA counts per model, new (base silent, model outputs) and fixed (base outputs, model silent);
- consistency: of one model's new FAs, how many the other models also produce (a shared cause vs decoding noise);
- what the new FA text is: share of its characters found in the interferers' transcripts (metadata.interferers) vs
  in nothing in the scene (hallucination), and its length;
- label bins of the new FAs: hard_cosine (enrollment vs interferer voice similarity), n_interferers, enroll seconds.
Usage (amphion-eval python): v12_fa_sources.py [--lang=en] SAMPLES.jsonl BASE_RUN NAME=RUN [NAME=RUN ...]
The last NAME is the one characterized in detail."""
import collections
import json
import sys

from amphion_eval.integrations.open_audio_llm import scoring

LANG = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--lang=")), "zh")
ARGS = [a for a in sys.argv[1:] if not a.startswith("--lang=")]


def read(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def outputs(run):
    """id -> normalized output tokens on target-absent zh TS samples."""
    out = {}
    for p in read(f"{run}/predictions.jsonl"):
        if p.get("task") == "ts_asr" and p.get("language") == LANG and not scoring.tokens(p.get("reference") or "", LANG):
            out[p["id"]] = scoring.tokens(p.get("prediction") or "", LANG)
    return out


def lcs_coverage(hyp, pool):
    """Share of hyp characters that appear, in order, in some interferer transcript (max over transcripts)."""
    best = 0
    for text in pool:
        ref = scoring.tokens(text, LANG)
        if not ref:
            continue
        prev = [0] * (len(ref) + 1)
        for h in hyp:
            cur = [0]
            for j, r in enumerate(ref):
                cur.append(prev[j] + 1 if h == r else max(prev[j + 1], cur[j]))
            prev = cur
        best = max(best, prev[-1])
    return best / len(hyp) if hyp else 0.0


def main():
    samples = {s["id"]: s for s in read(ARGS[0])}
    base = outputs(ARGS[1])
    models = [a.split("=", 1) for a in ARGS[2:]]
    outs = {name: outputs(run) for name, run in models}
    ids = sorted(base)
    for name in outs:
        assert set(outs[name]) == set(base), name
    fam = {i: str((samples[i]["record"].get("labels") or {}).get("scenario") or samples[i]["source"].split(":")[1].split("/")[0])
           for i in ids}
    group = {i: str((samples[i]["record"].get("labels") or {}).get("group")) for i in ids}
    print(f"absent samples {len(ids)}; base FA {sum(bool(base[i]) for i in ids)}")
    for name, o in outs.items():
        new = sum(bool(o[i]) and not base[i] for i in ids); fixed = sum(bool(base[i]) and not o[i] for i in ids)
        print(f"  {name:10s} FA {sum(bool(o[i]) for i in ids):4d}  new {new:4d}  fixed {fixed:4d}  net {new - fixed:+d}")

    names = list(outs)
    print("\nconsistency of each model's new FAs (also FA in the other models)")
    for name in names:
        new = [i for i in ids if outs[name][i] and not base[i]]
        others = [n for n in names if n != name]
        allo = sum(all(outs[n][i] for n in others) for i in new)
        anyo = sum(any(outs[n][i] for n in others) for i in new)
        print(f"  {name:10s} new {len(new):4d}  in all others {allo:4d}  in any other {anyo:4d}")
    fixed_all = [i for i in ids if base[i] and all(not outs[n][i] for n in names)]
    new_all = [i for i in ids if not base[i] and all(outs[n][i] for n in names)]
    print(f"  new in every model {len(new_all)}; fixed in every model {len(fixed_all)}")

    print("\nper group: base FA, then FA / new / fixed per model")
    for key, label in ((group, "group"), (fam, "scenario")):
        cells = collections.defaultdict(list)
        for i in ids:
            cells[key[i]].append(i)
        rows = []
        for cell, members in cells.items():
            b = sum(bool(base[i]) for i in members)
            per = [(sum(bool(outs[n][i]) for i in members), sum(bool(outs[n][i]) and not base[i] for i in members),
                    sum(bool(base[i]) and not outs[n][i] for i in members)) for n in names]
            rows.append((sum(p[0] for p in per) / len(per) - b, cell, len(members), b, per))
        rows.sort(reverse=True)
        print(f"-- by {label} (sorted by mean net change; top 25 and bottom 5)")
        show = rows if label == "group" else rows[:25] + [None] + rows[-5:]
        for r in show:
            if r is None:
                print("   ...")
                continue
            d, cell, n, b, per = r
            print(f"   {cell:38s} n {n:4d} base {b:3d}  " + "  ".join(f"{nm} {fa:3d}/+{nw}/-{fx}" for nm, (fa, nw, fx) in zip(names, per))
                  + f"  mean net {d:+.1f}")

    last = names[-1]; o = outs[last]
    new = [i for i in ids if o[i] and not base[i]]
    fixed = [i for i in ids if base[i] and not o[i]]
    print(f"\n{last}: new FA text")
    def describe(members, out):
        cov = []; lens = []
        for i in members:
            meta = samples[i]["record"].get("metadata") or {}
            pool = [u.get("text") or "" for it in (meta.get("interferers") or []) for u in (it.get("utts") or [])]
            cov.append(lcs_coverage(out[i], pool) if pool else None); lens.append(len(out[i]))
        with_pool = [c for c in cov if c is not None]
        hi = sum(c >= 0.6 for c in with_pool); lo = sum(c < 0.3 for c in with_pool)
        lens.sort()
        return (f"n {len(members)}; with interferer text {len(with_pool)}: coverage >=0.6 {hi}, <0.3 {lo}; "
                f"no interferer text in metadata {len(cov) - len(with_pool)}; output chars median {lens[len(lens) // 2] if lens else 0}, "
                f"<=2 chars {sum(x <= 2 for x in lens)}")
    print("  new  :", describe(new, o))
    print("  base FAs (base output):", describe([i for i in ids if base[i]], base))
    for lab, bins in (("hard_cosine", (0.0, 0.3, 0.4, 0.5, 0.6, 1.01)), ("enroll_seconds", (0, 3, 5, 8, 100)),
                      ("n_interferers", (0, 1, 2, 3, 100))):
        print(f"  {lab}: bin  n  base FA  {last} FA")
        for lo_, hi_ in zip(bins, bins[1:]):
            members = [i for i in ids if (v := (samples[i]["record"].get("labels") or {}).get(lab)) is not None and lo_ <= v < hi_]
            if members:
                print(f"    [{lo_}, {hi_})  {len(members):5d}  {sum(bool(base[i]) for i in members):4d}  {sum(bool(o[i]) for i in members):4d}")
    print(f"\n{last}: 25 new FA examples (scenario | output | interferer coverage)")
    for i in new[:: max(1, len(new) // 25)][:25]:
        meta = samples[i]["record"].get("metadata") or {}
        pool = [u.get("text") or "" for it in (meta.get("interferers") or []) for u in (it.get("utts") or [])]
        print(f"  {i:34s} {''.join(o[i])[:40]:42s} cov {lcs_coverage(o[i], pool) if pool else float('nan'):.2f}")


main()
