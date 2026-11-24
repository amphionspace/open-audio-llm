"""Split benchmark CER by whether the reference text also appears in a training source.

Usage: split_cer.py <hits.json> <label>=<predictions.jsonl> ...
hits.json: {"<normalized text>": ["<train source>", ...]} written by overlap_scan.py as <out>-hits.json.
"""
import collections
import difflib
import json
import re
import sys
import unicodedata

SRC_KEYS = ['aishell@', 'aishell3@', 'aishell2@', 'magicdata@', 'thchs30@',
            'wenetspeech@icefall-20260908:test_net', 'common_voice_zh@']


def norm(t):
    return re.sub(r'[\W_]+', '', unicodedata.normalize('NFKC', t).lower())


def errs(r, h):
    m = sum(b.size for b in difflib.SequenceMatcher(None, r, h, autojunk=False).get_matching_blocks())
    return max(len(r), len(h)) - m


hits = json.load(open(sys.argv[1]))
out = {}
for arg in sys.argv[2:]:
    label, path = arg.split('=', 1)
    acc = collections.defaultdict(lambda: [0, 0, 0, 0, 0])  # n_in, e_in, c_in, e_out, c_out
    for line in open(path):
        d = json.loads(line)
        k = next((k for k in SRC_KEYS if d['source'].startswith(k)), None)
        if k is None:
            continue
        r, h = norm(d['reference'] or ''), norm(d.get('prediction') or '')
        a = acc[k]
        if r in hits:
            a[0] += 1; a[1] += errs(r, h); a[2] += len(r)
        else:
            a[3] += errs(r, h); a[4] += len(r)
    out[label] = {k: {'in_train_utts': a[0],
                      'cer_in': round(100 * a[1] / max(a[2], 1), 2),
                      'cer_out': round(100 * a[3] / max(a[4], 1), 2)} for k, a in acc.items()}
print(json.dumps(out, ensure_ascii=False, indent=1))
