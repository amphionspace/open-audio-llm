"""跟错人 on ts-asr-first-other@v3 predictions, paired against a baseline run (sign test on the per-item flag).

Definition from oal-grpo docs/experiments/2026-11-06-grpo-v10-release: target present (pos) = the output covers <60% of
the target and >40% of the output is not target; target absent (neg) = any output. Coverage is the character LCS of
the punctuation-free lowercased texts. On the release run this reproduces the package's table exactly.
usage: follow_wrong.py <baseline run dir> <candidate run dir>
"""
import collections
import json
import math
import re
import sys


def norm(s):
    return re.sub(r"[\W_]", "", (s or "").lower())


def lcs_len(a, b):
    prev = [0] * (len(b) + 1)
    for ca in a:
        cur = [0]
        for j, cb in enumerate(b):
            cur.append(prev[j] + 1 if ca == cb else max(prev[j + 1], cur[j]))
        prev = cur
    return prev[-1]


def flags(run):
    out = {}
    for line in open(f"{run}/predictions.jsonl"):
        d = json.loads(line)
        split = d["id"].rsplit(":", 1)[0]
        stratum, kind = re.match(r"(.+?)-(?:alimeeting|aishell4)-.*-(pos|neg)$", split).groups()
        ref, hyp = norm(d["reference"]), norm(d.get("prediction"))
        if kind == "neg":
            bad = bool(hyp)
        else:
            m = lcs_len(ref, hyp)
            bad = m / len(ref) < 0.6 and bool(hyp) and (len(hyp) - m) / len(hyp) > 0.4
        out[d["id"]] = (f"{stratum}/{kind}", bad)
    return out


def sign_p(x, y):
    n, k = x + y, min(x, y)
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def main():
    a, b = flags(sys.argv[1]), flags(sys.argv[2])
    assert a.keys() == b.keys()
    rows = collections.defaultdict(lambda: [0, 0, 0, 0, 0])
    for i, (split, bad_a) in a.items():
        bad_b = b[i][1]
        r = rows[split]
        r[0] += 1; r[1] += bad_a; r[2] += bad_b; r[3] += bad_b and not bad_a; r[4] += bad_a and not bad_b
    print("split\tn\tbaseline%\tcandidate%\tnew\tfixed\tsign p")
    for split, (n, x, y, new, fixed) in sorted(rows.items()):
        print(f"{split}\t{n}\t{100 * x / n:.2f}\t{100 * y / n:.2f}\t{new}\t{fixed}\t{sign_p(new, fixed):.3g}")


if __name__ == "__main__":
    main()
