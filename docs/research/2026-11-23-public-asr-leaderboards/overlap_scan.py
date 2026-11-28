"""Exact normalized-text overlap: benchmark refs vs every text field of zh training sources.

Scans all *.jsonl.gz under each source's version dir (except audio/record indexes).
Usage: overlap_scan.py <predictions.jsonl> <data-effective.yaml> <out.json> [source-prefix,...]
predictions.jsonl may also be a run's samples.jsonl (references only). The optional fourth
argument replaces the default test-set source prefixes.
"""
import collections
import difflib
import glob
import gzip
import json
import re
import sys
import unicodedata
from multiprocessing import Pool

import yaml

PRED, DATA_CFG, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
DEFAULT_SRC_KEYS = ['aishell@', 'aishell3@', 'aishell2@', 'magicdata@', 'thchs30@',
            'wenetspeech@icefall-20260908:test_net', 'common_voice_zh@']
SRC_KEYS = sys.argv[4].split(',') if len(sys.argv) > 4 else DEFAULT_SRC_KEYS
EN_ONLY = {'librimix', 'gigaspeech', 'librispeech', 'common_voice_en', 'ami_sdm', 'chime6', 'fleurs_en'}
SKIP_KEYS = {'id', 'audio', 'path', 'cut_id', 'recording_id', 'source', 'version', 'dataset_id', 'split'}
MIN_LEN = 6  # shorter refs collide by chance


def norm(t):
    return re.sub(r'[\W_]+', '', unicodedata.normalize('NFKC', t).lower())


refs = collections.defaultdict(dict)
for line in open(PRED):
    d = json.loads(line)
    for k in SRC_KEYS:
        if d['source'].startswith(k):
            refs[k][d['id']] = (norm(d['reference'] or ''), norm(d.get('prediction') or ''))
REF = {t for v in refs.values() for t, _ in v.values() if len(t) >= MIN_LEN}


def strings(o):
    if isinstance(o, str):
        yield o
    elif isinstance(o, dict):
        for k, v in o.items():
            if k not in SKIP_KEYS:
                yield from strings(v)
    elif isinstance(o, list):
        for v in o:
            yield from strings(v)


def scan(args):
    name, vdir = args
    files = [f for f in glob.glob(f'{vdir}/**/*.jsonl.gz', recursive=True)
             if 'audio_index' not in f and 'record_index' not in f and 'duration' not in f]
    found, n = set(), 0
    for f in files:
        try:
            with gzip.open(f, 'rt') as fh:
                for line in fh:
                    n += 1
                    try:
                        o = json.loads(line)
                    except Exception:
                        continue
                    for s in strings(o):
                        if len(s) >= MIN_LEN and not s.isascii():
                            ns = norm(s)
                            if ns in REF:
                                found.add(ns)
        except Exception as e:
            print('ERR', f, e, file=sys.stderr)
    print('scanned', name, len(files), n, len(found), file=sys.stderr, flush=True)
    return name, found, len(files), n


def cer(v, ids):
    e = n = 0
    for i in ids:
        r, h = v[i]
        m = sum(b.size for b in difflib.SequenceMatcher(None, r, h, autojunk=False).get_matching_blocks())
        e += max(len(r), len(h)) - m
        n += len(r)
    return round(100 * e / max(n, 1), 3)


if __name__ == '__main__':
    cfg = yaml.safe_load(open(DATA_CFG))
    jobs = {}
    for s in cfg['train']:
        if s['dataset_id'] in EN_ONLY:
            continue
        jobs[f"{s['dataset_id']}@{s['version']}"] = f"/workspace/data/datasets/{s['dataset_id']}/versions/{s['version']}"
    with Pool(16) as p:
        results = p.map(scan, list(jobs.items()), chunksize=1)
    hits = collections.defaultdict(set)
    coverage = {}
    for name, found, nfiles, nlines in results:
        coverage[name] = {'files': nfiles, 'lines': nlines}
        for t in found:
            hits[t].add(name)
    res = {'sources_scanned': coverage}
    for k, v in refs.items():
        inn = [i for i, (r, _) in v.items() if r in hits]
        s_in = set(inn)
        out = [i for i in v if i not in s_in]
        res[k] = {'total': len(v), 'text_in_train': len(inn), 'frac': round(len(inn) / len(v), 4),
                  'approx_cer_in': cer(v, inn), 'approx_cer_out': cer(v, out), 'approx_cer_all': cer(v, list(v)),
                  'by_train_source': collections.Counter(s for i in inn for s in hits[v[i][0]]).most_common(8)}
    json.dump({t: sorted(s) for t, s in hits.items()}, open(OUT.replace('.json', '-hits.json'), 'w'), ensure_ascii=False)
    json.dump(res, open(OUT, 'w'), ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != 'sources_scanned'}, ensure_ascii=False, indent=1))
