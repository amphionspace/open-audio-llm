from settings import ROOT, RUNS, OUTPUT, SETTINGS, artifact, child
"""Frozen full-set cpCER and paired-meeting tcpCER, including parse failures."""
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import runpy

D = ROOT
E = artifact('evaluation')
BASE = RUNS / 'qwen3-asr-sot-speaker-events-20260922'
scoring = runpy.run_path(str(BASE / 'statistical-moss-comparison-500-20260922/score.py'))
clips = list(map(json.loads, (E / 'clips.jsonl').read_text().splitlines()))
ids = [r['id'] for r in clips]; candidates = {}; output_metrics = {}
for name in ('control', 'treatment', 'initial'):
    predictions = list(map(json.loads, (E / name / 'predictions.jsonl').read_text().splitlines()))
    mapped = {r['id']: r for r in predictions}
    assert len(mapped) == len(predictions) == 338 and set(mapped) == set(ids)
    scores = [scoring['score_one'](c, mapped[c['id']], 'qwen') for c in clips]
    candidates[name] = {r['id']: r for r in scores}
    (E / name / 'scores.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in scores))
    # A proxy is reported explicitly as such; repeated legitimate replies are
    # not automatically labeled hallucinations.
    repetition = []
    for r in predictions:
        text = scoring['normalize'](r['text'])
        grams = Counter(text[i:i+8] for i in range(max(0, len(text)-7)))
        repetition.append(sum(n-1 for n in grams.values()) / max(1, sum(grams.values())))
    output_metrics[name] = {'empty_output_rate': sum(not scoring['normalize'](r['text']) for r in predictions) / 338,
        'mean_repeated_8gram_fraction_proxy': sum(repetition) / 338,
        'hit_token_limit_count': sum(r['hit_token_limit'] for r in predictions),
        'timed_parse_failures': sum(not r['timed_available'] for r in scores)}
datasets = ('alimeeting', 'aishell4')
by_candidate = {name: {ds: scoring['aggregate']([r for r in rows.values() if r['dataset'] == ds],
    {k for k, r in rows.items() if r['timed_available']}) for ds in datasets} for name, rows in candidates.items()}
comparisons = {}
for left, right in [('treatment', 'control'), ('treatment', 'initial'), ('control', 'initial')]:
    comparison = {}
    for metric, error in [('cpCER', 'cp_errors'), ('tcpCER_5s', 'tcp_errors_5s'), ('tcpCER_1s', 'tcp_errors_1s')]:
        paired = [(candidates[left][i], candidates[right][i]) for i in ids]
        selected = paired if metric == 'cpCER' else [(a, b) for a, b in paired if a['timed_available'] and b['timed_available']]
        groups = {ds: defaultdict(lambda: [0, 0, 0]) for ds in datasets}
        for a, b in selected:
            assert a['reference_characters'] == b['reference_characters']
            v = groups[a['dataset']][a['meeting_id']]
            v[0] += a[error]; v[1] += b[error]; v[2] += a['reference_characters']
        stats = scoring['uncertainty']({ds: list(v.values()) for ds, v in groups.items()}) if all(groups.values()) else {}
        for s in stats.values():
            for source, destination in [('qwen', 'left'), ('moss', 'right'), ('qwen_ci95', 'left_ci95'),
                    ('moss_ci95', 'right_ci95'), ('difference_qwen_minus_moss', 'difference_left_minus_right')]:
                s[destination] = s.pop(source)
        comparison[metric] = {'paired_samples': len(selected), 'full_selected_set': len(selected) == len(ids),
            'statistics': stats, 'excluded_ids': [a['id'] for a, b in paired if metric != 'cpCER' and not (a['timed_available'] and b['timed_available'])]}
    comparisons[f'{left}_vs_{right}'] = comparison
primary = comparisons['treatment_vs_control']
def improved(metric):
    result = primary[metric]; macro = result['statistics'].get('equal_weight_corpus_macro', {})
    return (result['full_selected_set'] and macro.get('difference_left_minus_right', 1) < 0
        and macro.get('significant_at_005', False))
success = improved('tcpCER_5s') and improved('cpCER')
summary = {'samples': 338, 'meetings': 28, 'backend': 'vllm', 'training_steps_per_arm': 1000,
    'by_candidate': by_candidate, 'output_metrics': output_metrics, 'comparisons': comparisons,
    'supports_larger_followup': success,
    'decision': 'Clean-pipeline pilot supports a larger follow-up' if success else 'Do not expand training automatically; inspect paired effect size, uncertainty and failure coverage',
    'limitations': ['Previously used fixed development set, not a fresh confirmatory test.',
        'Joint data-and-generator intervention, not a separate attribution of each cause.',
        'Identity screen is experimental, not a calibrated identity error rate.',
        'Incomplete timed parsing is retained in full cpCER; tcpCER uses reported common coverage.']}
(E / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
with (E / 'paired-results.csv').open('w', newline='') as f:
    w = csv.writer(f); w.writerow(['id', 'dataset', 'meeting', 'arm', 'cpCER', 'tcpCER_5s', 'tcpCER_1s', 'timed_available', 'cap'])
    for i in ids:
        for name, rows in candidates.items():
            r = rows[i]; w.writerow([i, r['dataset'], r['meeting_id'], name, r['cpCER'], r.get('tcpCER_5s'), r.get('tcpCER_1s'), r['timed_available'], r['hit_token_limit']])
print(json.dumps({'decision': summary['decision'], 'primary': primary}, ensure_ascii=False), flush=True)
