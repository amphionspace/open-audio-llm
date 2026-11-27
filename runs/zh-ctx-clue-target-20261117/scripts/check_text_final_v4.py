"""Check the text-final v4 (Arabic digits, rule 9) versions of context_correction_synth_zh against what the ctx recipe
needs (README "数据登记"): (1) labels.context_correction (segments, cot.final) follow the converted target; (2) number
words that name a character in a spelling explanation ("一百的百": a numeral word, 的, then one of its own characters)
stay as written; stroke / radical descriptions ("三点水", "一横") stay as written; (3) no target keeps a protected
word's alternative ("反应|反映": the TTS read both spellings, so such records belong in quarantine).
usage: python3 -I check_text_final_v4.py"""
import collections, glob, gzip, json, re

V = "/workspace/data/datasets/context_correction_synth_zh/versions"
NUM = "零一二三四五六七八九十百千万两"
SELF = re.compile(rf"([{NUM}]+)的([{NUM}])")
STROKE = re.compile(rf"[{NUM}]+(点水|横|竖|撇|捺|画|笔|字头|字底|个口|个人|个木|个日|个土)")
c, examples = collections.Counter(), set()
for ver in ("train-v1-20261114", "implicit-v1-20261117", "ts-twins-v1-20261121", "asr-dialog-v1-20261122"):
    for f in sorted(glob.glob(f"{V}/{ver}-final-v4-20261127/text/*-records.jsonl.gz")):
        if "quarantine" in f:
            continue
        k = (ver, f.split("/")[-1].removesuffix("-records.jsonl.gz"))
        for line in gzip.open(f, "rt"):
            r = json.loads(line)
            cc = (r.get("labels") or {}).get("context_correction") or {}
            tgt = r["target"]
            orig = ((r.get("metadata") or {}).get("text_final") or {}).get("original_text") or tgt
            c[k, "records"] += 1
            c[k, "target with |"] += "|" in tgt
            if orig != tgt:
                c[k, "target changed"] += 1
                c[k, "segments == new target"] += "".join(s["text"] for s in cc.get("segments", [])) == tgt
                c[k, "cot.final == new target"] += (cc.get("cot") or {}).get("final") == tgt
            for m in SELF.finditer(orig):
                if m.group(2) in m.group(1):
                    kept = m.group(0) in tgt
                    c[k, f"self-naming numeral {'kept' if kept else 'converted'}"] += 1
                    if not kept:
                        examples.add(orig[max(0, m.start() - 10):m.end() + 6])
            for m in STROKE.finditer(orig):
                c[k, f"stroke description {'kept' if m.group(0) in tgt else 'converted'}"] += 1
missing = [v for v in ("train-v1-20261114", "implicit-v1-20261117", "ts-twins-v1-20261121", "asr-dialog-v1-20261122")
           if not glob.glob(f"{V}/{v}-final-v4-20261127/text/*-records.jsonl.gz")]
print("no v4 version:", missing or "none")
for k in sorted({x[0] for x in c}):
    print(*k, {n: v for (kk, n), v in c.items() if kk == k})
print("converted self-naming numerals:", *sorted(examples), sep="\n  ")
