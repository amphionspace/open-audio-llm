"""Mined_real items (127, WenetSpeech podcasts) whose target word or explanation (>= 4 chars) occurs in the explained
training data (think-on WenetSpeech transcripts, synthetic train/train_ts targets, every think quote), and the paired
changes on the items where neither occurs. usage: check_mined_overlap.py <output json of the seen item ids>"""
import gzip, json, re, glob, sys
AE = "/workspace/data/ae-runs"
R = {"release_off": "spelling-think-20261113-release-v10interp050/runs/run-20261114T100542Z-f99aafd9",
     "control_off": "spelling-think-20261113-control-3000/runs/run-20261114T121919Z-0f1385ac",
     "r2_on": "spelling-think-r2-20261115-think-3000-think/runs/run-20261115T072618Z-03dd8a4a",
     "avg3_off": "spelling-think-grpo-20261115-grpo-avg3/runs/run-20261115T122409Z-4bf6261c",
     "avg3_on": "spelling-think-grpo-20261115-grpo-avg3-think/runs/run-20261115T122649Z-e0b3a6b5"}
norm = lambda s: re.sub(r"[^一-鿿]", "", s)
ok = {k: {} for k in R}
for k, r in R.items():
    for l in open(f"{AE}/{r}/context-correction-scores.jsonl"):
        x = json.loads(l)
        if x["kind"] == "mined": ok[k][x["id"]] = x["target_ok"]
items = {}
for l in open(f"{AE}/{R['release_off']}/samples.jsonl"):
    x = json.loads(l)
    if x["id"] in ok["release_off"]:
        mt = x["record"]["labels"]["context_correction"]["marked_text"]
        items[x["id"]] = {"targets": [t.split("|")[0] for t in re.findall(r"\[\[T:([^\]]+)\]\]", mt)],
                          "evidence": [norm(d) for d in re.findall(r"\[\[D:([^\]]+)\]\]", mt)]}
# explained training texts: transcripts of think-on WenetSpeech records + synthetic train/train_ts targets
TH = "/workspace/data/datasets/context_correction_clean_zh/versions/think-v2-20261113"
quotes = []
want = set()
for f in glob.glob(f"{TH}/*/think.jsonl.gz"):
    for l in gzip.open(f, "rt"):
        d = json.loads(l)
        if d["think_on"] and d["think"]:
            quotes += [norm(q) for q in re.findall(r"「([^」]+)」，写作", d["think"])]
            if "wenetspeech_clean" in f: want.add(d["record_id"].rsplit(":", 1)[1])
texts = []
for l in gzip.open("/workspace/data/datasets/wenetspeech_clean/versions/train-punc-v1-20261102-ctxclean-20261110/supervisions/train.jsonl.gz", "rt"):
    d = json.loads(l)
    if d["id"] in want: texts.append(norm(d["text"]))
SY = "/workspace/data/datasets/context_correction_synth_zh/versions/train-v1-20261114"
for sp in ("train", "train_ts"):
    for l in gzip.open(f"{SY}/records/{sp}.jsonl.gz", "rt"):
        texts.append(norm(json.loads(l)["target"]))
for l in gzip.open(f"{SY}/think.jsonl.gz", "rt"):
    d = json.loads(l)
    if d.get("think"): quotes += [norm(q) for q in re.findall(r"「([^」]+)」，写作", d["think"])]
print("explained transcripts (WenetSpeech think-on + synthetic)", len(texts), "think quotes", len(quotes), file=sys.stderr)
blob = "\n".join(texts); qset = set(quotes)
seen = {}
for i, it in items.items():
    word = any(len(norm(t)) >= 2 and norm(t) in blob for t in it["targets"])
    expl = any(len(e) >= 4 and (e in blob or e in qset) for e in it["evidence"])
    seen[i] = word or expl
uns = [i for i in items if not seen[i]]
print("items", len(items), "target word or explanation seen in explained training data", len(items) - len(uns), "unseen", len(uns))
for k in R:
    print(f"{k:12s} all {sum(ok[k].values())}/127  unseen {sum(ok[k][i] for i in uns)}/{len(uns)}  seen {sum(ok[k][i] for i in items if seen[i])}/{len(items)-len(uns)}")
import math
def sign(a, b, ids):
    up = sum(1 for i in ids if not ok[a][i] and ok[b][i]); dn = sum(1 for i in ids if ok[a][i] and not ok[b][i]); n = up + dn
    p = min(1, 2 * sum(math.comb(n, j) for j in range(min(up, dn) + 1)) / 2 ** n) if n else 1
    return f"+{up}/-{dn} p={p:.4f}"
for a, b in [("release_off", "avg3_on"), ("avg3_off", "avg3_on"), ("control_off", "avg3_on")]:
    print(a, "->", b, "unseen", sign(a, b, uns))
json.dump({"seen": [i for i in items if seen[i]]}, open(sys.argv[1], "w"), ensure_ascii=False)
