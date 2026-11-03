"""AliMeeting dev windows from the gated headsets with a different "recording device" per speaker (shortcut probe).

Synthetic training meetings draw every talker from a different corpus and device (AISHELL-1 microphone, AISHELL-2
iPhone, MagicData Android, ...), so streams also differ by channel; in a real meeting everyone shares one room and
one microphone. Here each speaker of a meeting gets a distinct device-like filter (flat; 4 kHz low-pass; 200 Hz-6.5 kHz
band with a 3 kHz presence peak; brighter tilt; darker tilt), its level on its own turns kept, then the gated
headsets are summed exactly like near_gated. If heavy-overlap errors drop well below near_gated, the model separates
talkers by channel cues that real meetings do not have. Record ids "near_channel/<panel id>"."""
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
from scipy.signal import butter, iirpeak, lfilter, sosfilt

from amphiondata.multispeaker.farfield import activity, load_channel0, stable_int
from amphiondata.multispeaker.meetings import RATE

DBFS = -26.0


def device(kind, x):
    if kind == 'flat':
        return x
    if kind == 'lowpass4k':
        return sosfilt(butter(8, 4000, 'low', fs=RATE, output='sos'), x)
    if kind == 'band_presence':
        y = sosfilt(butter(4, [200, 6500], 'band', fs=RATE, output='sos'), x)
        b, a = iirpeak(3000, 1.5, fs=RATE)
        return y + 0.6 * lfilter(b, a, y)
    if kind == 'bright':
        return lfilter([1.0, -0.7], [1.0], x)
    if kind == 'dark':
        return 0.5 * x + 0.5 * sosfilt(butter(1, 1500, 'low', fs=RATE, output='sos'), x)
    raise ValueError(kind)


KINDS = ('flat', 'lowpass4k', 'band_presence', 'bright', 'dark')


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--panel-records', type=Path, required=True)
    p.add_argument('--near-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--catalog-dir', type=Path, required=True)
    args = p.parse_args()
    split = 'near_channel'
    (args.output / split).mkdir(parents=True, exist_ok=True)
    records = [json.loads(line) for line in gzip.open(args.panel_records, 'rt')]
    by_meeting = defaultdict(list)
    for r in records:
        by_meeting[r['metadata']['meeting_id']].append(r)
    out_records, index_rows, assignment = [], [], {}
    for meeting, rows in sorted(by_meeting.items()):
        spans = defaultdict(list)
        for r in rows:
            s0 = r['metadata']['source_record']['start']
            for t in r['metadata']['turns']:
                spans[t['source_speaker']].append((s0 + t['start'], s0 + t['end']))
        speakers = sorted(spans)
        order = np.random.default_rng(stable_int(20261031, meeting)).permutation(len(KINDS))
        mixed = None
        for i, spk in enumerate(speakers):
            (path,) = glob.glob(f'{args.near_dir}/{meeting}_N_SPK{spk}.wav')
            x = load_channel0(path)
            gate = activity(spans[spk], len(x), 0.25, 0)
            x = x * gate
            own = activity(spans[spk], len(x), 0, 0) > 0
            kind = KINDS[order[i % len(KINDS)]]
            y = device(kind, x).astype(np.float32)
            y *= np.sqrt(np.mean(x[own] ** 2) / (np.mean(y[own] ** 2) + 1e-12))
            assignment.setdefault(meeting, {})[spk] = kind
            mixed = y if mixed is None else (np.pad(mixed, (0, max(0, len(y) - len(mixed)))) +
                                             np.pad(y, (0, max(0, len(mixed) - len(y)))))
        for r in rows:
            s0 = r['metadata']['source_record']['start']
            n = round(r['audio_slots'][0]['ref']['duration'] * RATE)
            piece = mixed[round(s0 * RATE):round(s0 * RATE) + n]
            act = activity([(t['start'], t['end']) for t in r['metadata']['turns']], len(piece), 0, 0) > 0
            piece = np.clip(piece * (10 ** (DBFS / 20) / np.sqrt(np.mean(piece[act] ** 2) + 1e-12)), -0.99, 0.99)
            rid = r['id']
            sf.write(args.output / split / f'{rid}.wav', piece, RATE, subtype='PCM_16')
            row = json.loads(json.dumps(r))
            row['id'] = f'{split}/{rid}'
            row['audio_slots'][0]['ref'].update({'dataset_id': 'meeting_simreal_probe', 'version': 'v1-20261031',
                                                 'split': split, 'cut_id': f'{split}/{rid}', 'start': 0.0,
                                                 'duration': len(piece) / RATE})
            out_records.append(row)
            index_rows.append({'channels': 1, 'cut_id': f'{split}/{rid}', 'duration': len(piece) / RATE,
                               'num_frames': len(piece), 'relative_path': f'{split}/{rid}.wav',
                               'root_alias': 'meeting_simreal_probe', 'sample_rate': RATE})
        print(meeting, assignment[meeting], flush=True)
    cat = args.catalog_dir
    rec_path = cat / 'records' / f'{split}.jsonl.gz'
    with gzip.open(rec_path, 'wt') as f:
        for row in out_records:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    idx_path = cat / 'audio-index-near-channel.jsonl.gz'
    with gzip.open(idx_path, 'wt') as f:
        for row in index_rows:
            f.write(json.dumps(row) + '\n')
    entries = yaml.safe_load((cat / 'catalog.yaml').read_text())
    e = entries[0]
    e['artifacts'] = [a for a in e['artifacts'] if a['name'] not in (f'{split}-records', 'audio-index-near-channel')]
    e['artifacts'] += [
        {'name': 'audio-index-near-channel', 'kind': 'audio-index', 'root_alias': 'meeting_simreal_probe_catalog',
         'relative_path': idx_path.name, 'sha256': sha256_file(idx_path), 'metadata': {'record_count': len(index_rows)}},
        {'name': f'{split}-records', 'kind': 'audio-records', 'root_alias': 'meeting_simreal_probe_catalog',
         'relative_path': f'records/{split}.jsonl.gz', 'sha256': sha256_file(rec_path),
         'metadata': {'record_count': len(out_records)}}]
    e['splits'][split] = {'artifacts': {'records': [f'{split}-records'], 'audio_index': ['audio-index-near-channel']},
                          'statistics': {'records': len(out_records)}}
    (cat / 'catalog.yaml').write_text(yaml.safe_dump(entries, sort_keys=False, allow_unicode=True))
    (cat / 'near-channel-assignment.json').write_text(json.dumps(assignment, indent=1) + '\n')
    print('done', len(out_records), flush=True)


if __name__ == '__main__':
    main()
