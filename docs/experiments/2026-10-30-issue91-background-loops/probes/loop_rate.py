"""Loop/degeneration rate over an AmphionEval run's predictions.

loop  = same 1-4 char unit repeated 8+ times in a row (punctuation and spaces stripped)
length = decoder hit max_new_tokens instead of stopping
empty = no text at all
Per source and total, so a guard's effect on normal speech and on background-only
clips can be read apart.
"""
import json, re, sys, collections

LOOP = re.compile(r"(.{1,4}?)\1{7,}")
STRIP = re.compile(r"[\s，。？！、,.?!]")

def scan(path):
    tot = collections.Counter()
    per = collections.defaultdict(collections.Counter)
    for line in open(path):
        r = json.loads(line)
        src = r.get("source", "?").split(":")[0]
        text = r.get("prediction") or ""
        c = per[src]
        for d in (tot, c):
            d["n"] += 1
            if LOOP.search(STRIP.sub("", text)):
                d["loop"] += 1
            if r.get("finish_reason") == "length":
                d["length"] += 1
            if not text.strip():
                d["empty"] += 1
    return tot, per

def pct(c, k):
    return 100.0 * c[k] / c["n"] if c["n"] else 0.0

for path in sys.argv[1:]:
    tot, per = scan(path)
    print(f"\n== {path}")
    print(f"   total n={tot['n']:7d}  loop {tot['loop']:5d} ({pct(tot,'loop'):.3f}%)  "
          f"length {tot['length']:5d} ({pct(tot,'length'):.3f}%)  empty {tot['empty']:5d} ({pct(tot,'empty'):.3f}%)")
    for src, c in sorted(per.items(), key=lambda kv: -kv[1]["loop"]):
        if c["loop"] or c["length"]:
            print(f"   {src[:60]:60} n={c['n']:6d} loop {c['loop']:4d} ({pct(c,'loop'):.2f}%) length {c['length']:4d} ({pct(c,'length'):.2f}%)")
