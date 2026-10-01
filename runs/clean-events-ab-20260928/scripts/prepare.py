from settings import ROOT, RUNS, OUTPUT, SETTINGS, artifact, child
"""Expand real clean/aligned training sources without rerunning unchanged ASR."""
from collections import Counter, defaultdict
import gzip
import hashlib
import heapq
import json
from pathlib import Path
import sqlite3
import time

D = ROOT
BASE = Path(SETTINGS['parameters']['source_data'])
OLD = RUNS / 'identity-synthesis-20260927'
CLEAN = RUNS / 'source-clean-local-dual-20260923'
VERSION = 'dual-qwen-moss-v1-20260923'
DATASETS = ['aishell', 'aishell2', 'kespeech', 'commonvoice_en_clean', 'librispeech']

def save(name, value):
    path = artifact(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)

def rank(value):
    return int(hashlib.sha256(('20260928:' + value).encode()).hexdigest(), 16)

def main():
    started = time.monotonic()
    policy = json.loads((OLD / 'recipe-pilot.json').read_text())['turn_policy']
    # Whole, recorded acknowledgment utterances only; never cut these words out
    # of a longer sentence. Each language has its own vocabulary and quota.
    policy['response_texts']['zh'] += ['好的好的', '好好好', '对对对对', '是是是', '没问题', '嗯好的', '明白了']
    policy['response_texts']['en'] += ['all right', 'alright', 'oh yes', 'certainly', 'thank you', 'of course', 'i see', "that's right", 'very well', 'yes sir', 'yes indeed', 'absolutely']
    normalize = lambda t: ''.join(c for c in t.lower() if c.isalnum())
    replies = {l: {normalize(t) for t in ts} for l, ts in policy['response_texts'].items()}
    save('source-turn-policy.json', policy)
    feedback = json.loads((OLD / 'review/human-feedback.json').read_text())
    excluded_groups = set(feedback['quarantined_speaker_groups'])
    prior = json.loads((RUNS / 'synthesis-listening-20260924/review/human-final.json').read_text())
    excluded_ids = {s for pair in prior['same_speaker_constraints_rejected'] for s in pair['source_ids']}
    db = sqlite3.connect((CLEAN / 'clean-sources/eligibility.sqlite').resolve().as_uri() + '?mode=ro', uri=True)
    eligible = defaultdict(set)
    for dataset, source_id in db.execute('SELECT dataset,source_id FROM eligible'):
        eligible[dataset].add(source_id)
    def category(r):
        seconds = r['words'][-1]['end'] - r['words'][0]['start']
        return 'response' if normalize(r['text']) in replies[r['language']] and seconds <= 4 else 'long' if seconds > 6 else 'ordinary'
    stats = defaultdict(lambda: {'count': 0, 'seconds': 0., 'response': 0})
    for i, line in enumerate(gzip.open(BASE / 'word-pool.jsonl.gz', 'rt'), 1):
        r = json.loads(line)
        if (r['split'] != 'train' or r['source_id'] not in eligible[r['dataset_id']]
                or r['speaker'] in excluded_groups):
            continue
        seconds = r['words'][-1]['end'] - r['words'][0]['start']
        if not .08 <= seconds <= 20:
            continue
        s = stats[r['speaker']]
        s['count'] += 1; s['seconds'] += seconds; s['response'] += category(r) == 'response'
        if i % 200000 == 0:
            save('prepare-progress.json', {'stage': 'inventory', 'scanned': i, 'speakers': len(stats)})
    selected = {json.loads(line)['speaker'] for line in gzip.open(OLD / 'pilot-screened-pool.jsonl.gz', 'rt')}
    selection = {}
    for ds in DATASETS:
        # 450 s covers a 300 s dominant-role scene with the existing 1.5 reserve.
        long = sorted([s for s, v in stats.items() if s.startswith(ds + ':') and v['seconds'] >= 450], key=rank)[:64]
        response = sorted([s for s, v in stats.items() if s.startswith(ds + ':') and v['response'] and v['count'] >= 3], key=rank)[:64]
        selected.update(long + response)
        selection[ds] = {'long_reservoir_speakers': long, 'response_speakers': response}
    selected -= excluded_groups
    buckets = defaultdict(list)
    limits = {'ordinary': 192, 'long': 64, 'response': 32}
    for i, line in enumerate(gzip.open(BASE / 'word-pool.jsonl.gz', 'rt'), 1):
        r = json.loads(line)
        if r['speaker'] not in selected or r['split'] != 'train' or r['source_id'] not in eligible[r['dataset_id']]:
            continue
        if not .08 <= r['words'][-1]['end'] - r['words'][0]['start'] <= 20:
            continue
        identity = f"{r['dataset_id']}:{VERSION}:{r['source_id']}"
        if identity in excluded_ids:
            continue
        kind = category(r); heap = buckets[r['speaker'], kind]
        item = (-rank(r['id']), r['id'], r)
        if len(heap) < limits[kind]: heapq.heappush(heap, item)
        elif item > heap[0]: heapq.heapreplace(heap, item)
    rows = []; counts = Counter(); kinds = Counter(); duration = defaultdict(float)
    for key in sorted(buckets):
        for _, _, r in sorted(buckets[key], reverse=True):
            old, text, verdict = db.execute('SELECT old_version,source_text,clean FROM eligible WHERE dataset=? AND source_id=?', (r['dataset_id'], r['source_id'])).fetchone()
            assert r['version'] == old and r['text'] == text
            clean = json.loads(verdict); assert clean['pass'] is True
            r.update(upstream_version=old, version=VERSION, id=f"{r['dataset_id']}:{VERSION}:{r['source_id']}", upstream_clean_pass=True, source_clean=clean)
            rows.append(r); counts[r['dataset_id']] += 1; kinds[category(r)] += 1
            duration[r['speaker']] += r['words'][-1]['end'] - r['words'][0]['start']
    db.close()
    with gzip.open(artifact('expanded-clean-pool.jsonl.gz'), 'wt', compresslevel=1) as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False, separators=(',', ':')) + '\n')
    save('roots.json', json.loads((OLD / 'roots.json').read_text()))
    save('source-selection.json', selection)
    save('inventory.json', {'records': len(rows), 'speakers': len(duration), 'hours': sum(r['duration'] for r in rows) / 3600,
         'aligned_speech_hours': sum(duration.values()) / 3600, 'datasets': dict(counts), 'categories': dict(kinds),
         'speakers_at_least_seconds': {str(s): sum(v >= s for v in duration.values()) for s in [120, 300, 450, 600]},
         'source_pool': str(BASE / 'word-pool.jsonl.gz'), 'clean_version': VERSION,
         'reuse_policy': 'Existing vLLM Qwen+MOSS consensus passes, exactly unchanged source ID/version/text/audio interval; reuse original word alignment.',
         'excluded_identity_groups': sorted(excluded_groups), 'identity_screening': 'pending',
         'elapsed_seconds': time.monotonic() - started, 'human_review_required': False})
    print((artifact('inventory.json')).read_text(), flush=True)

if __name__ == '__main__': main()
