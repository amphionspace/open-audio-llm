"""Fine-tune labels = the reference text with 11.17's own punctuation: align the teacher output on the clean version
(punctuation removed) to the reference character by character, then put each punctuation mark after the reference
character its preceding teacher character aligned to. Teacher empty or > 50% character error: reference + '。'.
usage: punct_labels.py MANIFEST TEACHER_GLOB OUT_MANIFEST"""
import glob, json, re, sys
PUNCT = set("，。？！、；：,.?!;:")
teacher = {}
for f in glob.glob(sys.argv[2]):
    for l in open(f, encoding="utf-8"):
        o = json.loads(l); teacher[o["key"]] = o["teacher"]
def align(r, h):
    n, m = len(r), len(h); D = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1): D[i][0] = i
    for j in range(m + 1): D[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i][j] = min(D[i-1][j-1] + (r[i-1] != h[j-1]), D[i-1][j] + 1, D[i][j-1] + 1)
    i, j, mp = n, m, {}
    while i or j:
        if i and j and D[i][j] == D[i-1][j-1] + (r[i-1] != h[j-1]): mp[j-1] = i-1; i, j = i-1, j-1
        elif i and D[i][j] == D[i-1][j] + 1: i -= 1
        else: j -= 1
    return D[n][m], mp
stats = {"n": 0, "fallback": 0}
with open(sys.argv[3], "w", encoding="utf-8") as f:
    for l in open(sys.argv[1], encoding="utf-8"):
        r = json.loads(l); ref = re.sub(r"\s+", "", r["reference"]); t = teacher.get(r["key"]) or ""
        chars, marks = [], {}  # marks[j] = punctuation after teacher char j
        for ch in re.sub(r"\s+", "", t):
            if ch in PUNCT:
                if chars: marks.setdefault(len(chars) - 1, "")
                if chars: marks[len(chars) - 1] += ch
            else: chars.append(ch)
        stats["n"] += 1
        dist, mp = align(ref, chars) if chars else (len(ref), {})
        if not chars or dist > 0.5 * max(len(ref), 1):
            label = ref + "。"; stats["fallback"] += 1
        else:
            after = {}
            for j, m in marks.items():
                k = j
                while k >= 0 and k not in mp: k -= 1
                pos = mp[k] if k >= 0 else len(ref) - 1
                after[pos] = after.get(pos, "") + m[-1]
            label = "".join(c + after.get(i, "") for i, c in enumerate(ref))
            if label[-1] not in PUNCT: label += "。"
        r["label"] = label; f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(stats)
