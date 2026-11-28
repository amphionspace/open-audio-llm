"""tasks/30 language-ID localization: ar-asr-test runs of 11.17 and its module-swap models (swap_modules.py), scored
with the runs' own v6 rules. Per model and source: predicted_language counts, share of outputs in Arabic script, WER;
and WER on the same utterances where the first model wrote Arabic script (separates "lost the Arabic tag" from
"worse Arabic ASR once tagged"). usage: ar_lid_swap_report.py AE_SRC NAME=RUN_DIR ...   (amphion-eval python -I;
AE_SRC = the pinned amphion-eval with Arabic scoring, put first on sys.path because -I drops PYTHONPATH)"""
import collections, json, re, sys
sys.path.insert(0, sys.argv[1])
from amphion_eval.integrations.open_audio_llm import scoring

AR = re.compile(r"[؀-ۿ]")
runs = []
for arg in sys.argv[2:]:
    name, d = arg.split("=", 1); rows = {}
    for line in open(d + "/predictions.jsonl", encoding="utf-8"):
        p = json.loads(line)
        if p.get("status") != "success" or not scoring.presence_text(p["reference"]): continue
        c = scoring.row_counts(p, scoring.variety(p["language"]))
        src = p["source"].split("@")[0] + ":" + p["source"].split(":")[1].split("/")[0]
        rows[p["id"]] = (src, c["errors"], c["reference_units"], bool(AR.search(p.get("prediction") or "")), p.get("predicted_language"))
    runs.append((name, rows))
first = runs[0][1]
for name, rows in runs:
    print(f"== {name}  tags {dict(collections.Counter(r[4] for r in rows.values()).most_common(4))}")
    agg = collections.defaultdict(collections.Counter)
    for i, (src, e, n, ar, _) in rows.items():
        for k in (src, "ALL"):
            a = agg[k]; a["n"] += 1; a["ar"] += ar; a["e"] += e; a["r"] += n
            if first[i][3]: a["sub_n"] += 1; a["sub_e"] += e; a["sub_r"] += n
    for k, a in sorted(agg.items()):
        print(f"  {k:45s} n={a['n']:6d} ar_script={a['ar'] / a['n']:.2f} WER={100 * a['e'] / a['r']:6.1f}  "
              f"on {runs[0][0]}-Arabic-script n={a['sub_n']:5d} WER={100 * a['sub_e'] / max(1, a['sub_r']):6.1f}")
