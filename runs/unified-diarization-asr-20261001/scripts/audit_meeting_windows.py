"""Audit real-meeting training windows against the original human annotations.

For every derived window, compare the training target with the source supervisions
that fall inside the window audio: speech audible in the window but missing from the
target teaches the model to drop words; time or text drift teaches wrong labels.
"""
import argparse
import ast
import collections
import gzip
import json
from pathlib import Path
import unicodedata

import yaml


def jsonl(path):
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def norm(text):
    return ''.join(c for c in unicodedata.normalize('NFKC', text).lower()
                   if not c.isspace() and unicodedata.category(c)[0] not in {'P', 'S'})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    settings = yaml.safe_load(parser.parse_args().config.read_text())['parameters']
    output = Path(settings['output'])
    output.mkdir(parents=True, exist_ok=True)
    tolerance = settings['time_tolerance_seconds']
    minimum = settings['unlabeled_speech_min_seconds']
    summary, examples = {}, collections.defaultdict(list)
    for dataset, spec in settings['datasets'].items():
        supervisions = collections.defaultdict(list)
        for path in spec['supervisions']:
            for row in jsonl(path):
                supervisions[row['recording_id']].append(row)
        known = {row['id'] for rows in supervisions.values() for row in rows}
        for view, path in spec['views'].items():
            stats = collections.Counter()
            unlabeled_seconds = labeled_seconds = 0.0
            for record in jsonl(path):
                ref = record['audio_slots'][0]['ref']
                start, duration = ref['start'], ref['duration']
                turns = record['metadata']['turns']
                turns = ast.literal_eval(turns) if isinstance(turns, str) else turns
                stats['windows'] += 1
                if not turns:
                    stats['zero_speaker_windows'] += 1
                ids = {t['source_id'] for t in turns}
                stats['turns'] += len(turns)
                stats['turns_unmatched_source'] += sum(i not in known for i in ids)
                for t in turns:
                    if t['end'] > duration + 0.01 or t['start'] < -0.01:
                        stats['turns_outside_window'] += 1
                window_unlabeled = 0.0
                by_id = {}
                for s in supervisions[ref['cut_id']]:
                    s_start, s_end = s['start'] - start, s['start'] + s['duration'] - start
                    overlap = min(s_end, duration) - max(s_start, 0.0)
                    if overlap <= 0 or not norm(s.get('text') or ''):
                        continue
                    if s['id'] in ids:
                        by_id[s['id']] = (s_start, s_end, s['text'])
                        labeled_seconds += overlap
                    else:
                        window_unlabeled += overlap
                unlabeled_seconds += window_unlabeled
                if window_unlabeled >= minimum:
                    stats['windows_with_unlabeled_speech'] += 1
                    if not turns:
                        stats['zero_speaker_windows_with_speech'] += 1
                    if len(examples[f'{dataset}/{view}/unlabeled']) < 20:
                        examples[f'{dataset}/{view}/unlabeled'].append(
                            {'id': record['id'], 'unlabeled_seconds': round(window_unlabeled, 2)})
                for t in turns:
                    source = by_id.get(t['source_id'])
                    if source is None:
                        continue
                    if abs(t['start'] - source[0]) > tolerance or abs(t['end'] - source[1]) > tolerance:
                        stats['turn_time_mismatch'] += 1
                    if norm(t['text']) != norm(source[2]):
                        stats['turn_text_mismatch'] += 1
                        if len(examples[f'{dataset}/{view}/text']) < 20:
                            examples[f'{dataset}/{view}/text'].append(
                                {'id': record['id'], 'target': t['text'], 'source': source[2]})
            stats = dict(stats)
            stats['unlabeled_speech_seconds'] = round(unlabeled_seconds, 1)
            stats['labeled_speech_seconds'] = round(labeled_seconds, 1)
            stats['unlabeled_speech_fraction'] = round(
                unlabeled_seconds / max(unlabeled_seconds + labeled_seconds, 1e-9), 4)
            summary[f'{dataset}/{view}'] = stats
            print(dataset, view, json.dumps(stats), flush=True)
    (output / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n')
    (output / 'examples.json').write_text(json.dumps(examples, indent=2, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()
