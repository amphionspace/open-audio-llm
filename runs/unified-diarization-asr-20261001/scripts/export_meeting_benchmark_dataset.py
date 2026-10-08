"""Export the 180 s meeting benchmark as an audio-data-contract dataset.

Writes portable AudioRecords (task speaker_attributed_asr, one file per split), an
audio index relative to the dataset's own root alias and a dataset-catalog/2.0 entry, so
AmphionEval and open-audio-llm resolve the clips by ``dataset_id@version:split``.
Audio is hard-linked from the benchmark build, not copied. A Lhotse cut manifest per
split is also written for ``ae run --manifest``; it holds this machine's absolute
paths and is therefore not part of the catalog.
"""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path

import soundfile as sf
import yaml


def write_jsonl_gz(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    # mtime=0 keeps the gzip bytes, and so the catalog sha256, reproducible.
    with path.open('wb') as raw, gzip.GzipFile(fileobj=raw, mode='wb', mtime=0) as stream:
        for row in rows:
            stream.write((json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n').encode('utf-8'))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def target(reference):
    return '\n'.join(f"[{r['speaker']}][{r['start']:.2f}-{r['end']:.2f}] {r['text']}"
                     for r in sorted(reference, key=lambda r: (r['start'], r['end'])))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    settings = yaml.safe_load(parser.parse_args().config.read_text())['parameters']
    clips = [json.loads(line) for line in Path(settings['clips']).read_text().splitlines()]
    audio_dir = Path(settings['clip_audio_dir'])
    # The root alias points at the dataset directory, so a downloaded copy can live anywhere.
    data_root = Path(settings['dataset_root'])
    dataset_id, version = settings['dataset_id'], settings['version']
    source_dir = data_root / 'source' / version / 'audio'
    version_dir = data_root / 'versions' / version
    output = Path(settings['output'])
    output.mkdir(parents=True, exist_ok=True)
    source_dir.mkdir(parents=True, exist_ok=True)
    index, splits = [], {}
    for clip in clips:
        name = Path(clip['audio']).name
        original = audio_dir / name
        linked = source_dir / name
        if not linked.exists():
            os.link(original, linked)
        assert hashlib.sha256(linked.read_bytes()).hexdigest() == clip['audio_sha256'], name
        info = sf.info(str(linked))
        index.append({'cut_id': clip['id'], 'root_alias': settings['root_alias'],
                      'relative_path': str(linked.relative_to(data_root)), 'sample_rate': info.samplerate,
                      'channels': info.channels, 'num_frames': info.frames,
                      'duration': info.frames / info.samplerate, 'sha256': clip['audio_sha256']})
        split = clip['dataset']
        record = {
            'schema_version': 'audio-record/1.0', 'id': clip['id'], 'task': 'speaker_attributed_asr',
            'language': clip['language'],
            'audio_slots': [{'name': 'mixture', 'ref': {
                'dataset_id': dataset_id, 'version': version, 'split': split, 'cut_id': clip['id'],
                'start': 0.0, 'duration': info.frames / info.samplerate}}],
            'target': target(clip['reference']),
            'labels': {'speaker_count': clip['reference_speakers']},
            'metadata': {'sot_output_format': 'aligned_utterance_timestamps_v1',
                         'timestamp_source': 'human_meeting_annotation' if clip['kind'] == 'real'
                         else 'synthetic_timeline',
                         'turns': clip['reference'], 'kind': clip['kind'], 'meeting_id': clip['meeting_id'],
                         'overlap_seconds': clip['overlap_seconds'],
                         **{k: clip[k] for k in ('source_audio', 'source_start', 'snr_db') if k in clip}},
        }
        splits.setdefault(split, []).append(record)
    artifacts, split_specs = [], {}
    index_path = version_dir / 'audio-index.jsonl.gz'
    artifacts.append({'name': 'audio-index', 'kind': 'audio-index', 'root_alias': settings['root_alias'],
                      'relative_path': str(index_path.relative_to(data_root)),
                      'sha256': write_jsonl_gz(index_path, index),
                      'metadata': {'record_count': len(index)}})
    cuts_dir = output / 'cuts'
    for split, records in sorted(splits.items()):
        path = version_dir / 'records' / f'{split}.jsonl.gz'
        artifacts.append({'name': f'{split}-records', 'kind': 'audio-records',
                          'root_alias': settings['root_alias'], 'relative_path': str(path.relative_to(data_root)),
                          'sha256': write_jsonl_gz(path, records), 'metadata': {'record_count': len(records)}})
        hours = sum(r['audio_slots'][0]['ref']['duration'] for r in records) / 3600
        split_specs[split] = {'artifacts': {'records': [f'{split}-records'], 'audio_index': ['audio-index']},
                              'statistics': {'records': len(records), 'duration_hours': round(hours, 4),
                                             'duration_basis': 'Clip audio duration'}}
        by_id = {row['cut_id']: row for row in index}
        write_jsonl_gz(cuts_dir / f'{split}.jsonl.gz', (
            {'id': r['id'], 'start': 0.0, 'duration': r['audio_slots'][0]['ref']['duration'], 'channel': 0,
             'type': 'MonoCut',
             'recording': {'id': r['id'], 'sampling_rate': by_id[r['id']]['sample_rate'],
                           'num_samples': by_id[r['id']]['num_frames'],
                           'duration': by_id[r['id']]['duration'], 'channel_ids': [0],
                           'sources': [{'type': 'file', 'channels': [0],
                                        'source': str(data_root / by_id[r['id']]['relative_path'])}]},
             'supervisions': [{'id': f"{r['id']}-{i:04d}", 'recording_id': r['id'], 'start': t['start'],
                               'duration': round(t['end'] - t['start'], 3), 'channel': 0, 'text': t['text'],
                               'speaker': t['speaker'], 'language': r['language']}
                              for i, t in enumerate(r['metadata']['turns'])]}
            for r in records))
    entry = {'schema_version': 'dataset-catalog/2.0', 'dataset_id': dataset_id, 'version': version,
             'languages': sorted({c['language'] for c in clips}), 'tasks': ['speaker_attributed_asr'],
             'artifacts': artifacts, 'splits': split_specs, 'provenance': settings['provenance']}
    (output / 'catalog-entry.yaml').write_text(yaml.safe_dump([entry], allow_unicode=True, sort_keys=False))
    report = {'records': len(index), 'splits': {k: len(v) for k, v in splits.items()},
              'version_dir': str(version_dir), 'artifacts': {a['name']: a['sha256'] for a in artifacts}}
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report)[:600])


if __name__ == '__main__':
    main()
