from settings import ROOT, RUNS, OUTPUT, SETTINGS, artifact, child
"""Automatic identity quarantine using real AntSpeaker evidence and prior judgments."""
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path
import numpy as np
from prepare import save

D = ROOT
def main():
    rows = [json.loads(x) for x in gzip.open(artifact('expanded-clean-pool.jsonl.gz'), 'rt')]
    entries = {}; vectors = []; offset = 0
    for worker in range(2):
        array = np.load(artifact(f'embeddings-{worker}.npz'))['embeddings']
        for r in json.loads((artifact(f'inputs-{worker}.json')).read_text()):
            if r['status'] == 'ok': r['embedding_index'] += offset
            entries[r['id']] = r
        vectors.append(array); offset += len(array)
    matrix = np.concatenate(vectors)
    assert len(entries) == len(rows) and np.isfinite(matrix).all()
    groups = defaultdict(list)
    for r in rows:
        if entries[r['id']]['status'] == 'ok': groups[r['speaker']].append(r)
    allowed = {}; centroids = {}; audit = []; decisions = Counter()
    for speaker, group in groups.items():
        x = matrix[[entries[r['id']]['embedding_index'] for r in group]]
        sim = x @ x.T
        references = [i for i, r in enumerate(group) if entries[r['id']]['speech_seconds'] >= 2]
        cohort = []
        if len(references) >= 3:
            scores = sim[np.ix_(references, references)].copy(); np.fill_diagonal(scores, np.nan)
            medoid = references[int(np.nanargmax(np.nanmedian(scores, axis=1)))]
            cohort = [i for i in references if sim[medoid, i] >= .65]
        consistent = len(cohort) >= 3 and len(cohort) >= .6 * len(references)
        if consistent:
            v = x[cohort].mean(0); centroids[speaker] = v / np.linalg.norm(v)
        for i, r in enumerate(group):
            refs = [j for j in cohort if j != i]
            threshold = .55 if entries[r['id']]['speech_seconds'] < 2 else .65
            score = float(np.median(sim[i, refs])) if refs else None
            passed = consistent and len(refs) >= 2 and score >= threshold and (sim[i, refs] >= threshold).sum() >= 2
            decision = 'automatic_pass' if passed else 'identity_uncertain'
            decisions[decision] += 1
            evidence = {'status': decision, 'model': 'AntSpeaker-MECT-B2-vb2', 'backend': 'official_pytorch',
                'reference_median': score, 'threshold': threshold, 'threshold_calibrated': False,
                'reference_count': len(refs), 'cohort_fraction': len(cohort) / max(1, len(references)),
                'human_reviewed': False, 'evidence_file': str(artifact('identity-audit.jsonl'))}
            audit.append({'id': r['id'], 'speaker': speaker, **evidence})
            if passed:
                r['identity_screening'] = evidence; allowed[r['dataset_id'], r['source_id']] = r
    # Meeting crops already have independent ASR, purity, identity and alignment
    # results. Reuse them, preserving their source root and original evidence.
    meeting = RUNS / 'automatic-meeting-expansion-20260927'
    ix = {r['id']: r for r in json.loads((meeting / 'identity-inputs.json').read_text())}
    mx = np.load(meeting / 'embeddings.npz')['embeddings']
    for dec in map(json.loads, (meeting / 'source-screen.jsonl').read_text().splitlines()):
        if dec['decision'] == 'pass' and dec['speaker'] not in centroids:
            v = mx[[ix[k]['embedding_index'] for k in dec['reference_ids']]].mean(0)
            centroids[dec['speaker']] = v / np.linalg.norm(v)
    for r in map(json.loads, gzip.open(meeting / 'clean-word-pool.jsonl.gz', 'rt')):
        assert r['source_clean']['pass'] is True and r['speaker'] in centroids
        allowed[r['dataset_id'], r['source_id']] = r
    with gzip.open(artifact('word-pool.jsonl.gz'), 'wt', compresslevel=1) as f:
        for key, r in sorted(allowed.items()): f.write(json.dumps(r, ensure_ascii=False) + '\n')
    with (artifact('identity-audit.jsonl')).open('w') as f:
        for r in audit: f.write(json.dumps(r, ensure_ascii=False) + '\n')
    speakers = sorted({r['speaker'] for r in allowed.values()})
    np.savez_compressed(artifact('centroids.npz'), embeddings=np.stack([centroids[s] for s in speakers]))
    save('centroid-speakers.json', speakers)
    roots = json.loads((artifact('roots.json')).read_text()); roots['automatic_meeting_sources'] = str(meeting)
    save('roots.json', roots)
    durations = defaultdict(float); response = Counter()
    policy = json.loads((artifact('source-turn-policy.json')).read_text())
    normalize = lambda t: ''.join(c for c in t.lower() if c.isalnum())
    for r in allowed.values():
        durations[r['speaker']] += r['words'][-1]['end'] - r['words'][0]['start']
        if normalize(r['text']) in {normalize(t) for t in policy['response_texts'][r['language']]}: response[r['language']] += 1
    summary = {'records': len(allowed), 'speakers': len(speakers), 'hours': sum(r['duration'] for r in allowed.values()) / 3600,
        'identity_decisions': dict(decisions), 'responses_by_language': dict(response),
        'speakers_at_least_seconds': {str(s): sum(v >= s for v in durations.values()) for s in [120, 300, 450, 600]},
        'identity_threshold_calibrated': False, 'human_review_required': False,
        'identity_policy': 'Same-speaker medoid cohort >=60%; exclude self; >=2 peers; experimental cosine bands. Quarantine uncertain; preserve original sources.'}
    save('clean-pool-summary.json', summary); print(json.dumps(summary), flush=True)

if __name__ == '__main__': main()
