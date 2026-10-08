"""Reuse fixed meeting scoring and compare independent ASR regression outputs."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
import runpy

from rapidfuzz.distance import Levenshtein
import yaml

from open_audio_llm.data.qwen3_asr import native_language
from asr_normalize import normalize


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    settings = yaml.safe_load(args.config.read_text())['parameters']
    root = Path(settings['output'])
    scoring = runpy.run_path(settings['scoring_helpers'])
    clips = read_rows(root / 'clips.jsonl')
    primary = settings.get('primary_candidate', 'unified')
    candidates = {}
    predictions = {name: root / name / 'predictions.jsonl' for name in settings['models']}
    predictions.update(settings['cached_predictions'])
    for name, path in predictions.items():
        mapped = {r['id']: r for r in read_rows(path)}
        assert set(mapped) == {c['id'] for c in clips}
        rows = [scoring['score_one'](c, mapped[c['id']], 'qwen') for c in clips]
        candidates[name] = {r['id']: r for r in rows}
        (root / name).mkdir(exist_ok=True)
        (root / name / 'scores.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
    comparison = {}
    for baseline in predictions:
        if baseline == primary:
            continue
        metrics = {}
        for metric, error in [('cpCER', 'cp_errors'), ('tcpCER_5s', 'tcp_errors_5s'), ('tcpCER_1s', 'tcp_errors_1s')]:
            pairs = [(candidates[primary][c['id']], candidates[baseline][c['id']]) for c in clips]
            pairs = pairs if metric == 'cpCER' else [(a, b) for a, b in pairs if a['timed_available'] and b['timed_available']]
            groups = {ds: defaultdict(lambda: [0, 0, 0]) for ds in ('aishell4', 'alimeeting')}
            for a, b in pairs:
                assert a['reference_characters'] == b['reference_characters']
                totals = groups[a['dataset']][a['meeting_id']]
                totals[0] += a[error]; totals[1] += b[error]; totals[2] += a['reference_characters']
            stats = scoring['uncertainty']({ds: list(g.values()) for ds, g in groups.items()}) if all(groups.values()) else {}
            metrics[metric] = {'paired_samples': len(pairs), 'full_set': len(pairs) == len(clips), 'statistics': stats}
        comparison[f'{primary}_vs_{baseline}'] = metrics
    by_candidate = {name: {ds: scoring['aggregate']([r for r in rows.values() if r['dataset'] == ds],
                                                   {k for k, r in rows.items() if r['timed_available']})
                           for ds in ('aishell4', 'alimeeting')} for name, rows in candidates.items()}
    asr = {}
    asr_predictions = {name: root / name / 'asr-predictions.jsonl' for name in settings['models']}
    asr_predictions.update(settings.get('cached_asr_predictions', {}))
    reference_identity = None
    for candidate, path in asr_predictions.items():
        groups = defaultdict(list)
        rows = read_rows(path)
        identity = {(r['dataset_id'], r['id'], r['condition']):
                    (r['reference'], r['reference_language']) for r in rows}
        assert len(identity) == len(rows)
        if reference_identity is None:
            reference_identity = identity
        assert identity == reference_identity, f'ASR reference mismatch: {candidate}'
        for row in rows:
            groups[(row['dataset_id'], row['condition'])].append(row)
        metrics = {}
        for (dataset, condition), rows in groups.items():
            errors = units = 0
            for row in rows:
                chinese = native_language(row['reference_language']) == 'Chinese'
                # Remove only speaker/timing markup; keep all lexical output, including untagged text.
                text = re.sub(r'\[S\d+\]|\[-?\d+(?:\.\d+)?(?:--?\d+(?:\.\d+)?)?\]', '', row['text'])
                ref, hyp = normalize(row['reference']), normalize(text)
                ref, hyp = (list(ref.replace(' ', '')), list(hyp.replace(' ', ''))) if chinese else (ref.split(), hyp.split())
                errors += Levenshtein.distance(ref, hyp); units += len(ref)
            metrics[f'{dataset}/{condition}'] = {'samples': len(rows), 'metric': 'CER' if chinese else 'WER',
                                                'errors': errors, 'reference_units': units, 'error_rate': errors / units,
                                                'token_limit_hits': sum(r['hit_token_limit'] for r in rows)}
        asr[candidate] = metrics
    result = {'backend': 'vllm', 'training_step': settings.get('training_step', 1000),
              'primary_candidate': primary, 'samples': len(clips),
              'diarization': {'by_candidate': by_candidate, 'comparisons': comparison},
              'asr_regression': asr,
              'limitations': ['Fixed development set; per-comparison statistics do not adjust for repeated checkpoint looks.',
                              'ASR regression has 16 fixed examples per source; it is a small regression check.',
                              'ASR error rates remove speaker/timestamp markup; raw model outputs are retained.']}
    (root / 'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
