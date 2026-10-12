"""Score the 180 s meeting benchmark with AmphionEval's diarization protocol.

cpER counts every sample: speaker-attributed words are read leniently from the raw
output, so a malformed timestamp never removes words from the score. DER needs
complete timing and is reported on the samples whose output parses fully, together
with that coverage. Headline groups are macro averages over datasets; differences to
other candidates carry a paired meeting-cluster bootstrap interval.
"""
import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

import yaml
from amphion_eval.parsers import ParseError, parse_moss_transcribe_diarize
from amphion_eval.score import _cp_errors, diarization_error

SPEAKER = re.compile(r'\[S(\d+)\]')
TIME = re.compile(r'\[\d+(?:\.\d+)?(?:-\d+(?:\.\d+)?)?\]')
OURS_LINE = re.compile(r'^\[S(\d+)\]\[(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?)\]\s*(.*)$')
# A degenerate loop: one character 30+ times or a 1-10 character unit 20+ times in a row.
LOOP = re.compile(r'(.)\1{29,}|(.{2,10}?)\2{19,}', re.DOTALL)
PROTOCOL = 'diarization'
COLLAR = 0.25


def lexical(text):
    """Speaker -> words, keeping every word even when timestamps are malformed."""
    text = TIME.sub(' ', text)
    groups, speaker, previous = defaultdict(list), 'unattributed', 0
    for match in SPEAKER.finditer(text):
        groups[speaker].append(text[previous:match.start()])
        speaker, previous = f'S{int(match.group(1))}', match.end()
    groups[speaker].append(text[previous:])
    return {k: ' '.join(v).strip() for k, v in groups.items() if ' '.join(v).strip()}


def timed(text, fmt):
    if fmt == 'moss':
        return parse_moss_transcribe_diarize(text)
    segments = []
    for line in text.splitlines():
        if not line.strip():
            continue
        match = OURS_LINE.match(line.strip())
        if match is None:
            raise ParseError(f'unparsed line: {line[:60]!r}')
        start, end = float(match[2]), float(match[3])
        if end < start:
            raise ParseError('segment ends before it starts')
        segments.append({'speaker': f'S{int(match[1])}', 'start': start, 'end': end, 'text': match[4].strip()})
    return segments


def speaker_text(segments):
    groups = defaultdict(list)
    for s in sorted(segments, key=lambda s: (s['start'], s['end'])):
        groups[str(s['speaker'])].append(s['text'])
    return {k: ' '.join(v) for k, v in groups.items()}


def score_sample(clip, prediction, fmt):
    text = prediction.get('raw_text') if fmt == 'open_audio_llm' else prediction['text']
    text = text if text is not None else prediction.get('text', '')
    # Qwen3-ASR prefixes "language X<asr_text>"; the transcript follows the tag.
    text = text.split('<asr_text>', 1)[-1]
    reference = clip['reference']
    hypothesis = lexical(text)
    errors, units = _cp_errors(speaker_text(reference), hypothesis, clip['language'], PROTOCOL)
    row = {'id': clip['id'], 'dataset': clip['dataset'], 'language': clip['language'], 'kind': clip['kind'],
           'meeting_id': clip['meeting_id'], 'reference_speakers': clip['reference_speakers'],
           'predicted_speakers': len([k for k in hypothesis if k != 'unattributed']),
           'cp_errors': errors, 'units': units, 'hit_token_limit': bool(prediction.get('hit_token_limit')),
           'repetition_loop': bool(LOOP.search(TIME.sub('', text))),
           'timed': False}
    try:
        segments = timed(text, fmt)
        times = diarization_error(reference, segments, collar=COLLAR, duration=clip['duration'], region='reference')
        row.update(timed=True, **{f'{k}_seconds': v for k, v in times.items()})
    except (ParseError, ValueError) as exc:
        row['parse_error'] = str(exc)[:200]
    return row


def bucket(row):
    n = row['reference_speakers']
    if row['kind'] == 'synthetic':
        return f'synthetic_{n}spk'
    return 'real_1-2spk' if n <= 2 else 'real_3-4spk' if n <= 4 else 'real_5-7spk'


def aggregate(rows):
    units = sum(r['units'] for r in rows)
    timed_rows = [r for r in rows if r['timed']]
    scored = sum(r['scored_seconds'] for r in timed_rows)
    wrong = sum(r['missed_seconds'] + r['false_alarm_seconds'] + r['confusion_seconds'] for r in timed_rows)
    return {'samples': len(rows), 'cp_error_rate': sum(r['cp_errors'] for r in rows) / units if units else None,
            'der': wrong / scored if scored else None, 'timed_coverage': len(timed_rows) / len(rows) if rows else None,
            'speaker_count_accuracy': sum(r['predicted_speakers'] == r['reference_speakers'] for r in rows) / len(rows)
            if rows else None, 'hit_token_limit': sum(r['hit_token_limit'] for r in rows),
            'repetition_loops': sum(r['repetition_loop'] for r in rows),
            'cp_error_rate_without_loops': _rate([r for r in rows if not r['repetition_loop']])}


def _rate(rows):
    units = sum(r['units'] for r in rows)
    return sum(r['cp_errors'] for r in rows) / units if units else None


HEADLINES = {'real_zh': ['aishell4_test', 'alimeeting_test'], 'real_en': ['ami_test'],
             'synthetic_6-10spk': ['synthetic_zh', 'synthetic_en', 'synthetic_zh-en']}


def macro(rows_by_dataset, datasets):
    rates = []
    for d in datasets:
        rows = rows_by_dataset.get(d, [])
        units = sum(r['units'] for r in rows)
        if units:
            rates.append(sum(r['cp_errors'] for r in rows) / units)
    return sum(rates) / len(rates) if rates else None


def bootstrap(primary, other, datasets, seed=42, draws=10000):
    """Paired meeting-cluster bootstrap of the macro cpER difference (primary - other)."""
    meetings = {d: sorted({r['meeting_id'] for r in primary if r['dataset'] == d}) for d in datasets}
    def index(rows):
        out = defaultdict(lambda: [0, 0])
        for r in rows:
            out[r['meeting_id']][0] += r['cp_errors']
            out[r['meeting_id']][1] += r['units']
        return out
    a, b = index(primary), index(other)
    rng, diffs = random.Random(seed), []
    for _ in range(draws):
        rates_a, rates_b = [], []
        for d in datasets:
            pick = [rng.choice(meetings[d]) for _ in meetings[d]]
            ua = sum(a[m][1] for m in pick)
            rates_a.append(sum(a[m][0] for m in pick) / ua)
            rates_b.append(sum(b[m][0] for m in pick) / sum(b[m][1] for m in pick))
        diffs.append(sum(rates_a) / len(rates_a) - sum(rates_b) / len(rates_b))
    diffs.sort()
    return [diffs[int(0.025 * draws)], diffs[int(0.975 * draws)]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    settings = yaml.safe_load(parser.parse_args().config.read_text())['parameters']
    root = Path(settings['output'])
    clips = [json.loads(line) for line in (root / 'clips.jsonl').read_text().splitlines()]
    sources = {name: (root / name / 'predictions.jsonl', 'open_audio_llm') for name in settings['models']}
    for name, spec in (settings.get('cached_predictions') or {}).items():
        sources[name] = (Path(spec['path'] if isinstance(spec, dict) else spec),
                         spec.get('format', 'open_audio_llm') if isinstance(spec, dict) else 'open_audio_llm')
    scored = {}
    for name, (path, fmt) in sources.items():
        predictions = {r['id']: r for r in map(json.loads, path.read_text().splitlines())}
        assert set(predictions) == {c['id'] for c in clips}, f'{name}: predictions do not match the clips'
        rows = [score_sample(c, predictions[c['id']], fmt) for c in clips]
        scored[name] = rows
        (root / name).mkdir(exist_ok=True)
        (root / name / 'benchmark-scores.jsonl').write_text(
            ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
    primary = settings.get('primary_candidate', next(iter(settings['models'])))
    result = {'protocol': {'name': PROTOCOL, 'collar': COLLAR, 'region': 'reference',
                           'cp': 'all samples, lenient speaker-tag extraction', 'der': 'fully parsed samples'},
              'samples': len(clips), 'primary_candidate': primary, 'candidates': {}, 'comparisons': {}}
    for name, rows in scored.items():
        by_dataset = defaultdict(list)
        for r in rows:
            by_dataset[r['dataset']].append(r)
        groups = defaultdict(list)
        for r in rows:
            groups[bucket(r)].append(r)
        result['candidates'][name] = {
            'headline_cp_error_rate': {h: macro(by_dataset, ds) for h, ds in HEADLINES.items()},
            'by_dataset': {d: aggregate(v) for d, v in sorted(by_dataset.items())},
            'by_speakers': {b: aggregate(v) for b, v in sorted(groups.items())},
            'all': aggregate(rows)}
    for name, rows in scored.items():
        if name == primary:
            continue
        result['comparisons'][f'{primary}_vs_{name}'] = {
            h: {'difference': macro(_split(scored[primary]), ds) - macro(_split(rows), ds),
                'ci95': bootstrap(scored[primary], rows, ds)} for h, ds in HEADLINES.items()}
    result['limitations'] = [
        'Synthetic 6-10 speaker meetings use held-out test speakers but MUSAN noise also used in training noise augmentation.',
        'Real windows are cut at utterance gaps near 180 s; deployment cutting may split utterances.',
        'Differences are not corrected for repeated comparisons.']
    (root / 'benchmark-summary.json').write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({n: c['headline_cp_error_rate'] for n, c in result['candidates'].items()}, indent=2))


def _split(rows):
    out = defaultdict(list)
    for r in rows:
        out[r['dataset']].append(r)
    return out


if __name__ == '__main__':
    main()
