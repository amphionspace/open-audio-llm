"""Build the cleaned full-parameter training data config and a held-out selection set.

Outputs (under parameters.output):
  excluded-records.txt   meeting windows removed from training: quarantined meetings,
                         empty-target windows and every window of held-out meetings
  selection/clips.jsonl  held-out meeting clips in the fixed-set clip format
  selection/audio/*.wav  mean of all far-field channels, 16 kHz PCM16
  selection/plan.json    the fixed-set inference plan with this manifest's checksum
  catalog.jsonl          the training catalog plus the noise dataset entries
  train-data.yaml        the training data config
"""
import argparse
import ast
import gzip
import hashlib
import json
from pathlib import Path
import random
import re

import numpy as np
import soundfile as sf
import yaml


def jsonl(path):
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def turns_of(record):
    turns = record['metadata']['turns']
    return ast.literal_eval(turns) if isinstance(turns, str) else turns


def overlap_seconds(turns, duration, step=0.01):
    active = np.zeros(int(duration / step) + 1, dtype=np.int16)
    for speaker in {t['speaker'] for t in turns}:
        mask = np.zeros_like(active, dtype=bool)
        for t in turns:
            if t['speaker'] == speaker:
                mask[int(max(t['start'], 0) / step):int(min(t['end'], duration) / step)] = True
        active += mask
    return float((active >= 2).sum() * step)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    settings = yaml.safe_load(parser.parse_args().config.read_text())['parameters']
    output = Path(settings['output'])
    (output / 'selection' / 'audio').mkdir(parents=True, exist_ok=True)
    rng = random.Random(settings['seed'])
    roots = json.loads(Path(settings['roots']).read_text())
    excluded, clips, report = set(), [], {}
    for dataset, spec in settings['meetings'].items():
        index = {row['cut_id']: row for row in jsonl(spec['audio_index'])}
        meetings = sorted(index)
        holdout = set(rng.sample(meetings, spec['holdout_meetings']))
        views = {view: list(jsonl(path)) for view, path in spec['views'].items()}
        counts = {'quarantined': 0, 'empty_target': 0, 'holdout': 0}
        for view, records in views.items():
            for record in records:
                cut = record['audio_slots'][0]['ref']['cut_id']
                reason = ('quarantined' if cut in spec['quarantine_meetings'] else
                          'holdout' if cut in holdout else
                          'empty_target' if not turns_of(record) else None)
                if reason:
                    excluded.add(record['id'])
                    counts[reason] += 1
        candidates = [r for r in views[spec['selection_view']]
                      if r['audio_slots'][0]['ref']['cut_id'] in holdout and turns_of(r)
                      and r['audio_slots'][0]['ref']['duration'] <= spec['selection_max_duration']]
        chosen = []
        for meeting in sorted(holdout):
            pool = [r for r in candidates if r['audio_slots'][0]['ref']['cut_id'] == meeting]
            per = spec.get('selection_per_meeting')
            chosen += sorted(rng.sample(pool, min(per, len(pool))) if per else pool, key=lambda r: r['id'])
        for record in chosen:
            ref = record['audio_slots'][0]['ref']
            entry = index[ref['cut_id']]
            source = Path(roots[entry['root_alias']]) / entry['relative_path']
            info = sf.info(str(source))
            audio, rate = sf.read(str(source), start=int(round(ref['start'] * info.samplerate)),
                                  frames=int(round(ref['duration'] * info.samplerate)),
                                  dtype='float32', always_2d=True)
            assert rate == 16000
            path = output / 'selection' / 'audio' / f"{record['id']}.wav"
            sf.write(str(path), audio.mean(axis=1), rate, subtype='PCM_16')
            turns = turns_of(record)
            clips.append({
                'id': record['id'], 'dataset': spec['label'], 'split': 'train-holdout',
                'meeting_id': f"{spec['label']}:{ref['cut_id']}", 'source_id': record['id'],
                'duration': round(len(audio) / rate, 3),
                'reference': [{k: t[k] for k in ('speaker', 'start', 'end', 'text', 'source_id')}
                              for t in turns],
                'reference_speakers': len({t['speaker'] for t in turns}),
                'overlap_seconds': overlap_seconds(turns, ref['duration']),
                'source_start': ref['start'], 'source_audio': str(source),
                'channel_policy': 'mean of all original far-field channels',
                'audio': str(path),
                'audio_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            })
        report[dataset] = {'holdout_meetings': sorted(holdout), 'excluded_windows': counts,
                           'selection_clips': len(chosen)}
    (output / 'excluded-records.txt').write_text(''.join(f'{i}\n' for i in sorted(excluded)))
    manifest = output / 'selection' / 'clips.jsonl'
    manifest.write_text(''.join(json.dumps(c, ensure_ascii=False) + '\n' for c in clips))
    plan = json.loads(Path(settings['reference_plan']).read_text())
    plan.update(manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
                samples=len(clips), meetings=len({c['meeting_id'] for c in clips}),
                audio_hours=sum(c['duration'] for c in clips) / 3600, priority_ids=[],
                purpose='checkpoint selection on held-out training meetings; not the MOSS comparison set')
    (output / 'selection' / 'plan.json').write_text(json.dumps(plan, indent=2, ensure_ascii=False) + '\n')

    catalog = Path(settings['catalog']).read_text()
    for item in settings['noise_catalog_entries']:
        entries = yaml.safe_load(Path(item['registry']).read_text())
        entry = next(e for e in entries if e['dataset_id'] == item['dataset_id'] and e['version'] == item['version'])
        catalog += json.dumps(entry, ensure_ascii=False) + '\n'
    (output / 'catalog.jsonl').write_text(catalog)

    data = yaml.safe_load(Path(settings['source_data']).read_text())
    removed = [s for s in data['train'] if s['split'].endswith(tuple(settings['drop_split_suffixes']))]
    data['train'] = [s for s in data['train'] if s not in removed]
    # Keep the multi-speaker synthetic group's total weight so the task mix is unchanged.
    group = [s for s in data['train'] if s['dataset_id'] in settings['rescale_dataset_ids']
             and re.search(settings['rescale_split_pattern'], s['split'])]
    kept = sum(s['weight'] for s in group)
    scale = (kept + sum(s['weight'] for s in removed)) / kept
    for s in group:
        s['weight'] = s['weight'] * scale
    for s in data['train']:
        if s['dataset_id'] in settings['exclude_records_datasets']:
            s['exclude_records'] = str((output / 'excluded-records.txt').resolve())
    data['catalog'] = str((output / 'catalog.jsonl').resolve())
    data['augmentation'].update(settings['augmentation'])
    data['noise_sources'] = settings['noise_sources']
    data['batching'].update(settings['batching'])
    (output / 'train-data.yaml').write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True))
    report.update(removed_sources=[s['split'] for s in removed], synthetic_weight_scale=scale,
                  excluded_records=len(excluded), selection_clips=len(clips),
                  selection_hours=round(sum(c['duration'] for c in clips) / 3600, 3))
    (output / 'report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
