"""Build the independent meeting selection panel as an audio-data-contract dataset.

Every window of the configured official dev views is kept (no difficulty or
length filtering). NOTSOFAR records each meeting on several unsynchronized
devices; one device per meeting is kept, rotating through the sorted device
list by sorted meeting index, so all meetings and device types are covered
without evaluating the same conversation five times.

Window audio is cut from the source recording, far-field channels are averaged
and written as 16 kHz PCM16 mono under the dataset's own root alias. Records keep
the source window's target, turns and metadata unchanged and add the source
record identity, so a result can always be traced to the derived source view.
"""
import argparse
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

RATE = 16000


def read_jsonl(path):
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl_gz(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    # mtime=0 keeps the gzip bytes, and so the catalog sha256, reproducible.
    with path.open('wb') as raw, gzip.GzipFile(fileobj=raw, mode='wb', mtime=0) as stream:
        for row in rows:
            stream.write((json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n').encode('utf-8'))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select(records, policy):
    if policy == 'all':
        return records
    assert policy == 'one_device_per_meeting', policy
    meetings = defaultdict(lambda: defaultdict(list))
    for record in records:
        meetings[record['metadata']['meeting_id']][record['metadata']['recording_id']].append(record)
    chosen = []
    for index, (_, devices) in enumerate(sorted(meetings.items())):
        names = sorted(devices)
        chosen += devices[names[index % len(names)]]
    return chosen


def window(path, index_row, start, duration):
    info = sf.info(str(path))
    assert (info.samplerate, info.channels, info.frames) == (
        index_row['sample_rate'], index_row['channels'], index_row['num_frames']), path
    assert info.samplerate == RATE, path
    first = round(start * RATE)
    frames = round(duration * RATE)
    audio, _ = sf.read(str(path), start=first, frames=frames, dtype='float32', always_2d=True)
    assert len(audio) == frames, (path, start, duration)
    return audio.mean(axis=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    settings = yaml.safe_load(parser.parse_args().config.read_text())['parameters']
    data_root = Path(settings['dataset_root'])
    dataset_id, version, alias = settings['dataset_id'], settings['version'], settings['root_alias']
    audio_dir = data_root / 'source' / version / 'audio'
    version_dir = data_root / 'versions' / version
    output = Path(settings['output'])
    output.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)
    index, splits, report = [], {}, {}
    for split, source in settings['sources'].items():
        roots = {k: Path(v) for k, v in source['roots'].items()}
        audio_index = {row['cut_id']: row for row in read_jsonl(source['audio_index'])}
        records = select(read_jsonl(source['records']), source['selection'])
        rows = []
        for record in records:
            ref = record['audio_slots'][0]['ref']
            row = audio_index[ref['cut_id']]
            path = roots[row['root_alias']] / row['relative_path'].removeprefix(
                source.get('strip_prefix', {}).get(row['root_alias'], ''))
            name = f"{record['id']}.wav"
            target = audio_dir / name
            if not target.exists():
                sf.write(str(target), window(path, row, ref['start'], ref['duration']), RATE, subtype='PCM_16')
            info = sf.info(str(target))
            digest = hashlib.sha256(target.read_bytes()).hexdigest()
            index.append({'cut_id': record['id'], 'root_alias': alias,
                          'relative_path': str(target.relative_to(data_root)), 'sample_rate': info.samplerate,
                          'channels': info.channels, 'num_frames': info.frames,
                          'duration': info.frames / info.samplerate, 'sha256': digest})
            rows.append({**record,
                         'audio_slots': [{'name': 'mixture', 'ref': {
                             'dataset_id': dataset_id, 'version': version, 'split': split,
                             'cut_id': record['id'], 'start': 0.0, 'duration': info.frames / info.samplerate}}],
                         'metadata': {**record['metadata'], 'kind': 'real',
                                      'source_record': {**{k: ref[k] for k in ('dataset_id', 'version', 'split')},
                                                        'id': record['id'], 'cut_id': ref['cut_id'],
                                                        'start': ref['start']}}})
        splits[split] = rows
        counts = defaultdict(int)
        for r in rows:
            counts[r['labels']['speaker_count']] += 1
        report[split] = {'records': len(rows), 'meetings': len({r['metadata']['meeting_id'] for r in rows}),
                         'hours': sum(r['audio_slots'][0]['ref']['duration'] for r in rows) / 3600,
                         'max_seconds': max(r['audio_slots'][0]['ref']['duration'] for r in rows),
                         'speaker_counts': dict(sorted(counts.items())),
                         'devices': sorted({r['metadata'].get('recording_id', r['metadata']['meeting_id'])
                                            for r in rows})}
    artifacts, split_specs = [], {}
    index_path = version_dir / 'audio-index.jsonl.gz'
    artifacts.append({'name': 'audio-index', 'kind': 'audio-index', 'root_alias': alias,
                      'relative_path': str(index_path.relative_to(data_root)),
                      'sha256': write_jsonl_gz(index_path, index), 'metadata': {'record_count': len(index)}})
    for split, rows in splits.items():
        path = version_dir / 'records' / f'{split}.jsonl.gz'
        artifacts.append({'name': f'{split}-records', 'kind': 'audio-records', 'root_alias': alias,
                          'relative_path': str(path.relative_to(data_root)),
                          'sha256': write_jsonl_gz(path, rows), 'metadata': {'record_count': len(rows)}})
        split_specs[split] = {'artifacts': {'records': [f'{split}-records'], 'audio_index': ['audio-index']},
                              'statistics': {'records': len(rows), 'duration_hours': round(report[split]['hours'], 4),
                                             'duration_basis': 'Clip audio duration'}}
    entry = {'schema_version': 'dataset-catalog/2.0', 'dataset_id': dataset_id, 'version': version,
             'languages': sorted({r['language'] for rows in splits.values() for r in rows}),
             'tasks': ['speaker_attributed_asr'], 'artifacts': artifacts, 'splits': split_specs,
             'provenance': settings['provenance']}
    (output / 'catalog-entry.yaml').write_text(yaml.safe_dump([entry], allow_unicode=True, sort_keys=False))
    (output / 'roots.json').write_text(json.dumps({alias: str(data_root)}, indent=2) + '\n')
    report = {'records': len(index), 'splits': report, 'version_dir': str(version_dir),
              'artifacts': {a['name']: a['sha256'] for a in artifacts}}
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: {x: v[x] for x in ('records', 'meetings', 'hours')} for k, v in report['splits'].items()}))


if __name__ == '__main__':
    main()
