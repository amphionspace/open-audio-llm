"""AliMeeting dev selection windows with every turn made sequential (no overlap), from the near-field headsets.

Each reference turn of a panel window is cut from its own speaker's headset track ([start - pad, end + pad], the
track gated to that speaker's annotated turns like the training builder) and the turns are laid out one after
another in their original start order with ``gap`` s of silence, so no two speakers overlap. Text, speaker labels and
order are unchanged, times are shifted. Per-turn errors can then be compared with the same turns heard in overlap
(near_gated variant) to separate "overlap masks the turn" from "the turn itself is hard". Record ids are
"near_seq/<panel id>" like the other variants; the extra split is added to the variants catalog."""
import argparse
import glob
import gzip
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

from amphiondata.multispeaker.farfield import activity, load_channel0
from amphiondata.multispeaker.meetings import RATE

PAD, GAP, DBFS = 0.15, 0.3, -26.0


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--panel-records', type=Path, required=True)
    p.add_argument('--near-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--catalog-dir', type=Path, required=True)
    args = p.parse_args()
    split = 'near_seq'
    (args.output / split).mkdir(parents=True, exist_ok=True)
    records = [json.loads(l) for l in gzip.open(args.panel_records, 'rt')]
    by_meeting = defaultdict(list)
    for r in records:
        by_meeting[r['metadata']['meeting_id']].append(r)
    out_records, index_rows = [], []
    for meeting, rows in sorted(by_meeting.items()):
        turns_abs = defaultdict(list)
        for r in rows:
            s0 = r['metadata']['source_record']['start']
            for t in r['metadata']['turns']:
                turns_abs[t['source_speaker']].append((s0 + t['start'], s0 + t['end']))
        tracks = {}
        for spk, spans in turns_abs.items():
            (path,) = glob.glob(f'{args.near_dir}/{meeting}_N_SPK{spk}.wav')
            x = load_channel0(path)
            tracks[spk] = x * activity(spans, len(x), 0.25, 0)
        for r in rows:
            s0 = r['metadata']['source_record']['start']
            pieces, new_turns, cursor = [np.zeros(round(0.5 * RATE), np.float32)], [], 0.5
            for t in sorted(r['metadata']['turns'], key=lambda t: t['start']):
                x = tracks[t['source_speaker']]
                a = max(0, round((s0 + t['start'] - PAD) * RATE)); b = min(len(x), round((s0 + t['end'] + PAD) * RATE))
                seg = x[a:b]
                start = cursor + PAD
                new_turns.append({**t, 'start': round(start, 2), 'end': round(start + (t['end'] - t['start']), 2)})
                pieces += [seg, np.zeros(round(GAP * RATE), np.float32)]
                cursor += len(seg) / RATE + GAP
            audio = np.concatenate(pieces)
            act = activity([(u['start'], u['end']) for u in new_turns], len(audio), 0, 0) > 0
            audio = audio * (10 ** (DBFS / 20) / np.sqrt(np.mean(audio[act] ** 2) + 1e-12))
            audio = np.clip(audio, -0.99, 0.99)
            rid = r['id']
            sf.write(args.output / split / f'{rid}.wav', audio, RATE, subtype='PCM_16')
            row = json.loads(json.dumps(r))
            row['id'] = f'{split}/{rid}'
            row['audio_slots'][0]['ref'].update({'dataset_id': 'meeting_simreal_probe', 'version': 'v1-20261031',
                                                 'split': split, 'cut_id': f'{split}/{rid}', 'start': 0.0,
                                                 'duration': round(len(audio) / RATE, 3)})
            row['metadata']['turns'] = new_turns
            row['target'] = '\n'.join(f"[{u['speaker']}][{u['start']:.2f}-{u['end']:.2f}] {u['text']}" for u in new_turns)
            out_records.append(row)
            index_rows.append({'channels': 1, 'cut_id': f'{split}/{rid}', 'duration': len(audio) / RATE,
                               'num_frames': len(audio), 'relative_path': f'{split}/{rid}.wav',
                               'root_alias': 'meeting_simreal_probe', 'sample_rate': RATE})
        print(meeting, len(rows), flush=True)
    cat = args.catalog_dir
    rec_path = cat / 'records' / f'{split}.jsonl.gz'
    with gzip.open(rec_path, 'wt') as f:
        for row in out_records:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    idx_path = cat / 'audio-index-near-seq.jsonl.gz'
    with gzip.open(idx_path, 'wt') as f:
        for row in index_rows:
            f.write(json.dumps(row) + '\n')
    entries = yaml.safe_load((cat / 'catalog.yaml').read_text())
    e = entries[0]
    e['artifacts'] = [a for a in e['artifacts'] if a['name'] not in (f'{split}-records', 'audio-index-near-seq')]
    e['artifacts'] += [
        {'name': 'audio-index-near-seq', 'kind': 'audio-index', 'root_alias': 'meeting_simreal_probe_catalog',
         'relative_path': idx_path.name, 'sha256': sha256_file(idx_path), 'metadata': {'record_count': len(index_rows)}},
        {'name': f'{split}-records', 'kind': 'audio-records', 'root_alias': 'meeting_simreal_probe_catalog',
         'relative_path': f'records/{split}.jsonl.gz', 'sha256': sha256_file(rec_path),
         'metadata': {'record_count': len(out_records)}}]
    e['splits'][split] = {'artifacts': {'records': [f'{split}-records'], 'audio_index': ['audio-index-near-seq']},
                          'statistics': {'records': len(out_records)}}
    (cat / 'catalog.yaml').write_text(yaml.safe_dump(entries, sort_keys=False, allow_unicode=True))
    print('done', len(out_records), flush=True)


if __name__ == '__main__':
    main()
