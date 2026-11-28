"""tasks/29: plain-ASR error rate on the "clean subset" of each test set (test records whose sentence does not appear
in the training sources; test-side lists from the data side, open-audio-llm PR #175 rule), next to the full set.
Per-utterance edits with the run's own scoring module (shared amphion-eval env, v3), and a check that the full-set
sum reproduces summary.json's errors. usage: clean_subset_report.py LISTS_DIR NAME=RUN_DIR ...   (amphion-eval python -I)
LISTS_DIR holds <dataset>@<version>@<split>.txt, one record id per line (ids as in samples.jsonl)."""
import collections, glob, json, os, sys
from amphion_eval.integrations.open_audio_llm import scoring
from rapidfuzz.distance import Levenshtein

seen = {}
for f in glob.glob(os.path.join(sys.argv[1], "*.txt")):
    key = os.path.basename(f)[:-4]  # dataset@version@split
    seen[key] = {l.strip() for l in open(f) if l.strip()}
print("lists:", {k: len(v) for k, v in sorted(seen.items())})
rows = {}
for arg in sys.argv[2:]:
    name, run = arg.split("=", 1)
    summ = json.load(open(f"{run}/summary.json"))["metrics"]
    acc = collections.defaultdict(lambda: [0, 0, 0, 0])  # full errors, full ref, clean errors, clean ref
    for o in map(json.loads, open(f"{run}/predictions.jsonl")):
        if o.get("task") != "asr" or o.get("language") != "Chinese": continue
        src = o["source"].split("/")[0]  # dataset@version:split
        key = src.replace(":", "@", 1)
        if key not in seen: continue
        ref = scoring.tokens(o.get("reference") or "", "zh"); hyp = scoring.tokens(o.get("prediction") or "", "zh")
        e = Levenshtein.distance(ref, hyp); a = acc[(src, o["source"])]
        a[0] += e; a[1] += len(ref)
        if o["id"] not in seen[key]: a[2] += e; a[3] += len(ref)
    rows[name] = acc
    for (src, full), a in sorted(acc.items()):
        m = summ.get(full)
        if m and m["errors"] != a[0]:
            print(f"  WARN {name} {full}: recomputed errors {a[0]} vs summary {m['errors']}")
names = list(rows)
print(f"\n{'set':45s} " + "  ".join(f"{n:>13s}" for n in names) + "   (full / clean CER %)")
keys = sorted({k for r in rows.values() for k in r})
tot = {n: [0, 0, 0, 0] for n in names}
for k in keys:
    cells = []
    for n in names:
        a = rows[n].get(k, [0, 0, 0, 0]); cells.append(f"{100*a[0]/max(1,a[1]):5.2f}/{100*a[2]/max(1,a[3]):5.2f}")
        for j in range(4): tot[n][j] += a[j]
    print(f"{k[0]:45s} " + "  ".join(f"{c:>13s}" for c in cells))
print(f"{'ALL these sets':45s} " + "  ".join(f"{100*t[0]/t[1]:5.2f}/{100*t[2]/t[3]:5.2f}" for t in tot.values()))
