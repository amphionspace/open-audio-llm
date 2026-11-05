"""Compare the MOSS enrollment-concat run with the AmphionASR baseline run on the same samples.

Scores every sample with AmphionEval's own normalization (scoring.tokens / presence_text),
pools them into the eval set's scene groups, and reports, for MOSS, a second target rule
(the speaker who talks most inside the enrollment) and how often the S01 assumption holds.
Run with the amphion-eval interpreter:  python compare.py --config <task yaml>
"""
import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

import yaml
from rapidfuzz.distance import Levenshtein
from amphion_eval.integrations.open_audio_llm.scoring import presence_text, tokens, variety

SEGMENT = re.compile(r"\s*\[(?P<start>\d+(?:\.\d+)?)\]\[(?P<speaker>S\d+)\](?P<text>.*?)\[(?P<end>\d+(?:\.\d+)?)\]",
                     re.DOTALL)


PARTIAL = re.compile(r"\s*\[(?P<start>\d+(?:\.\d+)?)\]\[(?P<speaker>S\d+)\](?P<text>.*)$", re.DOTALL)


def parse(text):
    """Same reading as the worker: complete segments, then a truncated tail with an open end."""
    segments, position = [], 0
    while position < len(text):
        match = SEGMENT.match(text, position)
        if match is None:
            break
        segments.append((match['speaker'], float(match['start']), float(match['end']), match['text'].strip()))
        position = match.end()
    tail = PARTIAL.match(text[position:])
    if tail:
        segments.append((tail['speaker'], float(tail['start']), float('inf'), tail['text'].strip()))
    return segments


def score(row, prediction):
    kind = variety(row['language'])
    if presence_text(row['reference']):
        ref = tokens(row['reference'], kind)
        return {'pos': 1, 'errors': Levenshtein.distance(ref, tokens(prediction, kind)), 'units': len(ref),
                'miss': int(not presence_text(prediction)), 'fa': 0}
    return {'pos': 0, 'errors': 0, 'units': 0, 'miss': 0, 'fa': int(bool(presence_text(prediction)))}


def pooled(items):
    pos = [i for i in items if i['pos']]
    neg = [i for i in items if not i['pos']]
    units = sum(i['units'] for i in pos)
    return {'n_pos': len(pos), 'n_neg': len(neg),
            'cer': 100 * sum(i['errors'] for i in pos) / units if units else None,
            'miss': 100 * sum(i['miss'] for i in pos) / len(pos) if pos else None,
            'fa': 100 * sum(i['fa'] for i in neg) / len(neg) if neg else None}


def bootstrap(a, b, metric, reps, seed):
    rng = random.Random(seed)
    n, diffs = len(a), []
    for _ in range(reps):
        idx = [rng.randrange(n) for _ in range(n)]
        diffs.append(pooled([b[i] for i in idx])[metric] - pooled([a[i] for i in idx])[metric])
    diffs.sort()
    return [round(diffs[int(0.025 * reps)], 2), round(diffs[int(0.975 * reps) - 1], 2)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    settings = yaml.safe_load(parser.parse_args().config.read_text())['parameters']
    base_dir, moss_dir, out = Path(settings['baseline_run']), Path(settings['moss_run']), Path(settings['output'])
    out.mkdir(parents=True, exist_ok=True)
    enroll, gap = settings['enroll_seconds'], settings['gap_seconds']
    boundary = enroll + gap / 2
    load = lambda d: {r['id']: r for r in map(json.loads, (d / 'predictions.jsonl').read_text().splitlines())}
    base, moss = load(base_dir), load(moss_dir)
    ids = [i for i in moss if moss[i]['status'] == 'success']
    missing = [i for i in moss if i not in base]
    assert not missing, f'{len(missing)} MOSS samples absent from the baseline run'
    failed = [i for i in base if i in moss and base[i]['status'] != 'success']
    ids = [i for i in ids if i not in set(failed)]

    rows, diag = {}, Counter()
    for i in ids:
        row = moss[i]
        segments = parse(row['raw_text'])
        talk = Counter()
        for speaker, start, end, _ in segments:
            talk[speaker] += max(0.0, min(end, enroll) - max(start, 0.0))
        enroll_speakers = [s for s, t in talk.items() if t > 0]
        main_speaker = talk.most_common(1)[0][0] if enroll_speakers else None
        alt = ' '.join(t for s, st, _, t in segments if s == main_speaker and st >= boundary)
        diag['samples'] += 1
        diag['no_speaker_in_enrollment'] += not enroll_speakers
        diag['several_speakers_in_enrollment'] += len(enroll_speakers) > 1
        diag['enrollment_speaker_not_S01'] += bool(enroll_speakers) and main_speaker != 'S01'
        diag['segment_spans_enrollment_and_mixture'] += any(st < boundary and en > enroll + gap for _, st, en, _ in segments)
        diag['decode_truncated'] += bool(segments) and segments[-1][2] == float('inf')
        label = row['source'].split(':', 1)[1].split('/')[0]
        rows[i] = {'label': label, 'base': score(row, base[i]['prediction']), 's01': score(row, row['prediction']),
                   'alt': score(row, alt)}

    systems = ('base', 's01', 'alt')
    base_settings = json.loads((base_dir / 'settings.json').read_text())
    groups = []
    for group in base_settings['data']['groups']:
        members = [r for r in rows.values() if r['label'] in set(group['labels'])]
        if members:
            groups.append({'section': group['section'], 'name': group['name'],
                           **{s: pooled([m[s] for m in members]) for s in systems}})
    order = list(rows.values())
    overall = {s: pooled([r[s] for r in order]) for s in systems}
    pos = [r for r in order if r['base']['pos']]
    neg = [r for r in order if not r['base']['pos']]
    reps, seed = settings['bootstrap_reps'], settings['seed']
    intervals = {s: {'cer': bootstrap([r['base'] for r in pos], [r[s] for r in pos], 'cer', reps, seed),
                     'miss': bootstrap([r['base'] for r in pos], [r[s] for r in pos], 'miss', reps, seed),
                     'fa': bootstrap([r['base'] for r in neg], [r[s] for r in neg], 'fa', reps, seed)}
                 for s in ('s01', 'alt')}
    result = {'samples': len(order),
              'excluded_failed': {'moss': sum(r['status'] != 'success' for r in moss.values()),
                                  'baseline': len(failed)},
              'overall': overall, 'difference_ci95_vs_base': intervals,
              'moss_diagnostics': dict(diag), 'groups': groups,
              'baseline_run': str(base_dir), 'moss_run': str(moss_dir)}
    (out / 'comparison.json').write_text(json.dumps(result, ensure_ascii=False, indent=1) + '\n')

    fmt = lambda v: '—' if v is None else f'{v:.1f}'
    lines = ['| 场景组 | 指标 | 054avg3 | MOSS·S01 | MOSS·注册主讲人 |', '|---|---|---:|---:|---:|']
    for g in groups:
        metrics = [('CER', 'cer'), ('漏识%', 'miss')] if g['base']['n_pos'] else [('误输出%', 'fa')]
        for title, key in metrics:
            lines.append(f"| {g['name']}（{g['base']['n_pos'] or g['base']['n_neg']} 条） | {title} | "
                         + ' | '.join(fmt(g[s][key]) for s in systems) + ' |')
    (out / 'groups.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({k: result[k] for k in ('samples', 'excluded_failed', 'overall', 'difference_ci95_vs_base',
                                             'moss_diagnostics')}, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
