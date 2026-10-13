"""Cut AliMeeting Eval single-speaker segments in two recording conditions.

A segment is a human-annotated utterance with no other speaker's utterance within
``margin`` seconds on either side, so the clip holds one talker; the padding stops at any
neighbouring utterance of the same talker, so the clip holds only the reference text. Each one is cut twice
on the same timeline: the far-field array (mean of the 8 channels, as in the selection
panel) and the talker's own headset. Both copies get the same one-speaker SOT reference,
so every model is scored by the same meeting scorer on identical text.
"""
import argparse
import gzip
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

DATASET, VERSION = 'alimeeting_eval_single_speaker', 'v1-20261013'


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, 'wt') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    p = yaml.safe_load(parser.parse_args().config.read_text())['parameters']
    out = Path(p['output'])
    with gzip.open(p['supervisions'], 'rt') as stream:
        sups = [json.loads(line) for line in stream]
    by_recording = defaultdict(list)
    for s in sups:
        by_recording[s['recording_id']].append(s)

    index, records = [], {'far': [], 'near': []}
    skipped = defaultdict(int)
    for recording, items in sorted(by_recording.items()):
        far_path = next(Path(p['far_dir']).glob(f'{recording}_MS*.wav'))
        far, rate = sf.read(far_path, dtype='float32')
        far = far.mean(axis=1)
        for s in items:
            start, end = s['start'], s['start'] + s['duration']
            text = s['text'].strip()
            if not p['min_seconds'] <= s['duration'] <= p['max_seconds'] or len(text) < p['min_chars']:
                skipped['length'] += 1
                continue
            others = [o for o in items if o is not s]
            if any(o['speaker'] != s['speaker'] and o['start'] < end + p['margin']
                   and start - p['margin'] < o['start'] + o['duration'] for o in others):
                skipped['other_speaker_nearby'] += 1
                continue
            if any(o['start'] < end and start < o['start'] + o['duration'] for o in others):
                skipped['overlaps_own_utterance'] += 1
                continue
            # The pad stops at any neighbouring utterance (the same talker's), so the clip holds only this text.
            lo = max([0.0, start - p['pad']] + [o['start'] + o['duration'] for o in others if o['start'] + o['duration'] <= start])
            hi = min([end + p['pad']] + [o['start'] for o in others if o['start'] >= end])
            near_path = Path(p['near_dir']) / f"{recording}_N_SPK{s['speaker']}.wav"
            near, near_rate = sf.read(near_path, dtype='float32', start=int(lo * rate), stop=int(hi * rate))
            assert near_rate == rate == p['sampling_rate']
            clips = {'far': far[int(lo * rate):int(hi * rate)], 'near': near}
            for condition, audio in clips.items():
                cut = f"{condition}-{s['id']}"
                wav = out / 'audio' / condition / f'{cut}.wav'
                wav.parent.mkdir(parents=True, exist_ok=True)
                sf.write(wav, audio, rate, subtype='PCM_16')
                duration = round(len(audio) / rate, 3)
                index.append({'cut_id': cut, 'root_alias': DATASET, 'relative_path': str(wav.relative_to(out)),
                              'sample_rate': rate, 'channels': 1, 'num_frames': len(audio), 'duration': duration})
                rel_start, rel_end = round(start - lo, 2), round(min(end, hi) - lo, 2)
                records[condition].append({
                    'schema_version': 'audio-record/1.0', 'id': cut, 'task': 'speaker_attributed_asr',
                    'language': 'zh', 'target': f'[S1][{rel_start:.2f}-{rel_end:.2f}] {text}',
                    'audio_slots': [{'name': 'mixture', 'ref': {'dataset_id': DATASET, 'version': VERSION,
                                                                'split': condition, 'cut_id': cut, 'start': 0.0,
                                                                'duration': duration}}],
                    'labels': {'speaker_count': 1},
                    'metadata': {'kind': 'real', 'meeting_id': recording, 'source_id': s['id'],
                                 'source_speaker': s['speaker'], 'condition': condition,
                                 'sot_output_format': 'aligned_utterance_timestamps_v1',
                                 'turns': [{'start': rel_start, 'end': rel_end, 'speaker': 'S1', 'text': text,
                                            'source_id': s['id'], 'source_speaker': s['speaker']}]}})

    write_jsonl(out / 'audio-index.jsonl.gz', index)
    artifacts = [{'name': 'audio-index', 'kind': 'audio-index', 'root_alias': DATASET,
                  'relative_path': 'audio-index.jsonl.gz', 'sha256': sha256(out / 'audio-index.jsonl.gz'),
                  'metadata': {'record_count': len(index)}}]
    splits = {}
    for condition, rows in records.items():
        path = out / 'records' / f'{condition}.jsonl.gz'
        write_jsonl(path, rows)
        artifacts.append({'name': f'{condition}-records', 'kind': 'audio-records', 'root_alias': DATASET,
                          'relative_path': str(path.relative_to(out)), 'sha256': sha256(path),
                          'metadata': {'record_count': len(rows)}})
        hours = sum(r['audio_slots'][0]['ref']['duration'] for r in rows) / 3600
        splits[condition] = {'artifacts': {'records': [f'{condition}-records'], 'audio_index': ['audio-index']},
                             'group': 'test', 'statistics': {'records': len(rows), 'duration_hours': round(hours, 4)},
                             'provenance': {'training_use': 'held_out_not_for_training'}}
    catalog = [{'schema_version': 'dataset-catalog/2.0', 'dataset_id': DATASET, 'version': VERSION,
                'languages': ['zh'], 'tasks': ['speaker_attributed_asr'], 'artifacts': artifacts, 'splits': splits,
                'provenance': {'source': 'alimeeting_sdm@icefall-20260908 dev supervisions; AliMeeting Eval far/near audio',
                               'description': __doc__.strip()}}]
    (out / 'catalog.yaml').write_text(yaml.safe_dump(catalog, sort_keys=False, allow_unicode=True))
    (out / 'roots.json').write_text(json.dumps({DATASET: str(out)}, indent=2) + '\n')
    speakers = {r['metadata']['source_speaker'] for r in records['far']}
    summary = {'segments': len(records['far']), 'speakers': len(speakers), 'skipped': dict(skipped),
               'hours_per_condition': splits['far']['statistics']['duration_hours'],
               'supervisions': len(sups)}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
