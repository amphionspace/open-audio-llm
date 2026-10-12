"""Expose an existing clip manifest to AmphionEval as a local audio-data-contract dataset.

Used for the consistency check only: the clips, audio and references that the
experiment scripts scored are written as AudioRecords (one split per clip
``dataset``) and an audio index whose root alias points at the existing audio
directory, so no audio is copied or re-encoded. The catalog stays in this
attempt; it is not a shared registration.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

import soundfile as sf
import yaml

TIMESTAMP_FORMAT = 'aligned_utterance_timestamps_v1'


def write_jsonl_gz(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('wb') as raw, gzip.GzipFile(fileobj=raw, mode='wb', mtime=0) as stream:
        for row in rows:
            stream.write((json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n').encode('utf-8'))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    settings = yaml.safe_load(parser.parse_args().config.read_text())['parameters']
    output = Path(settings['output'])
    clips = [json.loads(line) for line in Path(settings['clips']).read_text().splitlines()]
    audio_dir = Path(settings['clip_audio_dir'])
    dataset_id, version = settings['dataset_id'], settings['version']
    index, splits = [], {}
    for clip in clips:
        path = audio_dir / Path(clip['audio']).name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == clip['audio_sha256'], path
        info = sf.info(str(path))
        duration = info.frames / info.samplerate
        index.append({'cut_id': clip['id'], 'root_alias': settings['audio_alias'], 'relative_path': path.name,
                      'sample_rate': info.samplerate, 'channels': info.channels, 'num_frames': info.frames,
                      'duration': duration, 'sha256': clip['audio_sha256']})
        turns = sorted(clip['reference'], key=lambda r: (r['start'], r['end']))
        split = settings['splits'][clip['dataset']]
        splits.setdefault(split, []).append({
            'schema_version': 'audio-record/1.0', 'id': clip['id'], 'task': 'speaker_attributed_asr',
            'language': clip.get('language', settings['language']),
            'audio_slots': [{'name': 'mixture', 'ref': {'dataset_id': dataset_id, 'version': version, 'split': split,
                                                        'cut_id': clip['id'], 'start': 0.0, 'duration': duration}}],
            'target': '\n'.join(f"[{t['speaker']}][{t['start']:.2f}-{t['end']:.2f}] {t['text']}" for t in turns),
            'labels': {'speaker_count': clip['reference_speakers']},
            'metadata': {'sot_output_format': TIMESTAMP_FORMAT, 'turns': clip['reference'],
                         'meeting_id': clip['meeting_id'], 'source_clip_dataset': clip['dataset']}})
    artifacts = [{'name': 'audio-index', 'kind': 'audio-index', 'root_alias': settings['records_alias'],
                  'relative_path': 'audio-index.jsonl.gz',
                  'sha256': write_jsonl_gz(output / 'audio-index.jsonl.gz', index)}]
    split_specs = {}
    for split, records in sorted(splits.items()):
        artifacts.append({'name': f'{split}-records', 'kind': 'audio-records', 'root_alias': settings['records_alias'],
                          'relative_path': f'records/{split}.jsonl.gz',
                          'sha256': write_jsonl_gz(output / 'records' / f'{split}.jsonl.gz', records)})
        split_specs[split] = {'artifacts': {'records': [f'{split}-records'], 'audio_index': ['audio-index']}}
    entry = {'schema_version': 'dataset-catalog/2.0', 'dataset_id': dataset_id, 'version': version,
             'languages': sorted({r['language'] for v in splits.values() for r in v}),
             'tasks': ['speaker_attributed_asr'], 'artifacts': artifacts, 'splits': split_specs,
             'provenance': {'description': settings['description']}}
    (output / 'catalog.yaml').write_text(yaml.safe_dump([entry], allow_unicode=True, sort_keys=False))
    (output / 'roots.json').write_text(json.dumps(
        {settings['records_alias']: str(output), settings['audio_alias']: str(audio_dir)}, indent=2) + '\n')
    report = {'records': len(index), 'splits': {k: len(v) for k, v in splits.items()}}
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
