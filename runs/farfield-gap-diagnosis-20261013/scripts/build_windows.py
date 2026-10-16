"""Cut AliMeeting Eval far-field windows of several lengths from the same annotation windows.

The records are the meeting-long-v1 SOT windows (cut at pauses, human utterance times) that the
selection panel's 120 s split was built from; every window is cut from the far-field array as the
mean of its 8 channels, as in the panel, so window length is the only difference between splits.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

import soundfile as sf
import yaml

DATASET, VERSION = 'alimeeting_eval_windows', 'v1-20261013'


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, 'wt') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    p = yaml.safe_load(parser.parse_args().config.read_text())['parameters']
    out = Path(p['output'])
    with gzip.open(p['audio_index'], 'rt') as stream:
        sources = {row['cut_id']: Path(p['audio_root']) / row['relative_path'].removeprefix(p['strip_prefix'])
                   for row in map(json.loads, stream)}
    index, artifacts, splits, cache = [], [], {}, {}
    for split, path in p['records'].items():
        with gzip.open(path, 'rt') as stream:
            rows = [json.loads(line) for line in stream]
        out_rows = []
        for row in rows:
            ref = row['audio_slots'][0]['ref']
            if ref['cut_id'] not in cache:
                cache.clear()  # one meeting in memory at a time; records are grouped by meeting
                audio, rate = sf.read(sources[ref['cut_id']], dtype='float32')
                cache[ref['cut_id']] = (audio.mean(axis=1), rate)
            audio, rate = cache[ref['cut_id']]
            clip = audio[int(ref['start'] * rate):int((ref['start'] + ref['duration']) * rate)]
            wav = out / 'audio' / split / f"{row['id']}.wav"
            wav.parent.mkdir(parents=True, exist_ok=True)
            sf.write(wav, clip, rate, subtype='PCM_16')
            duration = round(len(clip) / rate, 3)
            index.append({'cut_id': row['id'], 'root_alias': DATASET, 'relative_path': str(wav.relative_to(out)),
                          'sample_rate': rate, 'channels': 1, 'num_frames': len(clip), 'duration': duration})
            row = {**row, 'audio_slots': [{'name': 'mixture', 'ref': {'dataset_id': DATASET, 'version': VERSION,
                                                                      'split': split, 'cut_id': row['id'],
                                                                      'start': 0.0, 'duration': duration}}]}
            out_rows.append(row)
        records = out / 'records' / f'{split}.jsonl.gz'
        write_jsonl(records, out_rows)
        artifacts.append({'name': f'{split}-records', 'kind': 'audio-records', 'root_alias': DATASET,
                          'relative_path': str(records.relative_to(out)),
                          'sha256': hashlib.sha256(records.read_bytes()).hexdigest(),
                          'metadata': {'record_count': len(out_rows)}})
        hours = sum(r['audio_slots'][0]['ref']['duration'] for r in out_rows) / 3600
        splits[split] = {'artifacts': {'records': [f'{split}-records'], 'audio_index': ['audio-index']},
                         'group': 'test', 'statistics': {'records': len(out_rows), 'duration_hours': round(hours, 4)},
                         'provenance': {'training_use': 'held_out_not_for_training'}}
    write_jsonl(out / 'audio-index.jsonl.gz', index)
    artifacts.insert(0, {'name': 'audio-index', 'kind': 'audio-index', 'root_alias': DATASET,
                         'relative_path': 'audio-index.jsonl.gz',
                         'sha256': hashlib.sha256((out / 'audio-index.jsonl.gz').read_bytes()).hexdigest(),
                         'metadata': {'record_count': len(index)}})
    catalog = [{'schema_version': 'dataset-catalog/2.0', 'dataset_id': DATASET, 'version': VERSION,
                'languages': ['zh'], 'tasks': ['speaker_attributed_asr'], 'artifacts': artifacts, 'splits': splits,
                'provenance': {'source': 'alimeeting_sdm_meeting_sot@meeting-long-v1-20260920 dev windows',
                               'description': __doc__.strip()}}]
    (out / 'catalog.yaml').write_text(yaml.safe_dump(catalog, sort_keys=False, allow_unicode=True))
    (out / 'roots.json').write_text(json.dumps({DATASET: str(out)}, indent=2) + '\n')
    summary = {split: s['statistics'] for split, s in splits.items()}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
