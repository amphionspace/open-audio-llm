"""AliMeeting dev selection windows whose turns are swapped for the same speaker's other turns (coherence probe).

Every reference turn of a panel window is replaced by a turn of the same speaker from elsewhere in the same meeting
(outside the window, closest duration, each used once per window), cut from the speaker's gated headset track and
placed at the original turn's start. Voices, levels, headset acoustics and the overlap layout stay (up to the
duration mismatch); what overlaps with what no longer belongs to one exchange. Compared with near_gated per overlap
category this isolates the effect of conversational coherence between overlapping talkers. Record ids
"near_shuffled/<panel id>"; the split is added to the variants catalog."""
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

PAD, DBFS = 0.1, -26.0


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--panel-records', type=Path, required=True)
    p.add_argument('--near-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--catalog-dir', type=Path, required=True)
    args = p.parse_args()
    split = 'near_shuffled'
    (args.output / split).mkdir(parents=True, exist_ok=True)
    records = [json.loads(line) for line in gzip.open(args.panel_records, 'rt')]
    by_meeting = defaultdict(list)
    for r in records:
        by_meeting[r['metadata']['meeting_id']].append(r)
    out_records, index_rows, mismatch = [], [], []
    for meeting, rows in sorted(by_meeting.items()):
        pool = defaultdict(list)  # speaker -> [(window id, abs start, abs end, turn)]
        for r in rows:
            s0 = r['metadata']['source_record']['start']
            for t in r['metadata']['turns']:
                pool[t['source_speaker']].append((r['id'], s0 + t['start'], s0 + t['end'], t))
        tracks = {}
        for spk, items in pool.items():
            (path,) = glob.glob(f'{args.near_dir}/{meeting}_N_SPK{spk}.wav')
            x = load_channel0(path)
            tracks[spk] = x * activity([(a, b) for _, a, b, _ in items], len(x), 0.25, 0)
        for r in rows:
            duration = r['audio_slots'][0]['ref']['duration']
            frames = round(duration * RATE)
            audio = np.zeros(frames + 20 * RATE, np.float32)
            used, new_turns = set(), []
            for t in sorted(r['metadata']['turns'], key=lambda t: t['start']):
                spk, d = t['source_speaker'], t['end'] - t['start']
                cands = [c for c in pool[spk] if c[0] != r['id'] and (c[0], c[3]['source_id']) not in used]
                if not cands:
                    cands = [c for c in pool[spk] if c[0] != r['id']]
                w, a, b, u = min(cands, key=lambda c: abs((c[2] - c[1]) - d))
                used.add((w, u['source_id']))
                mismatch.append((b - a) - d)
                seg = tracks[spk][max(0, round((a - PAD) * RATE)):round((b + PAD) * RATE)]
                at = max(0, round((t['start'] - PAD) * RATE))
                audio[at:at + len(seg)] += seg[:len(audio) - at]
                new_turns.append({**u, 'speaker': t['speaker'], 'start': round(t['start'], 2),
                                  'end': round(t['start'] + (b - a), 2)})
            end = max([frames] + [round(u['end'] * RATE) for u in new_turns])
            audio = audio[:end]
            act = activity([(u['start'], u['end']) for u in new_turns], len(audio), 0, 0) > 0
            audio = np.clip(audio * (10 ** (DBFS / 20) / np.sqrt(np.mean(audio[act] ** 2) + 1e-12)), -0.99, 0.99)
            rid = r['id']
            sf.write(args.output / split / f'{rid}.wav', audio, RATE, subtype='PCM_16')
            row = json.loads(json.dumps(r))
            row['id'] = f'{split}/{rid}'
            row['audio_slots'][0]['ref'].update({'dataset_id': 'meeting_simreal_probe', 'version': 'v1-20261031',
                                                 'split': split, 'cut_id': f'{split}/{rid}', 'start': 0.0,
                                                 'duration': round(len(audio) / RATE, 3)})
            new_turns.sort(key=lambda u: u['start'])
            row['metadata']['turns'] = new_turns
            row['target'] = '\n'.join(f"[{u['speaker']}][{u['start']:.2f}-{u['end']:.2f}] {u['text']}" for u in new_turns)
            out_records.append(row)
            index_rows.append({'channels': 1, 'cut_id': f'{split}/{rid}', 'duration': len(audio) / RATE,
                               'num_frames': len(audio), 'relative_path': f'{split}/{rid}.wav',
                               'root_alias': 'meeting_simreal_probe', 'sample_rate': RATE})
        print(meeting, len(rows), flush=True)
    m = np.abs(np.array(mismatch))
    print(f'duration mismatch |d| median {np.median(m):.2f} s, p90 {np.percentile(m, 90):.2f} s', flush=True)
    cat = args.catalog_dir
    rec_path = cat / 'records' / f'{split}.jsonl.gz'
    with gzip.open(rec_path, 'wt') as f:
        for row in out_records:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    idx_path = cat / 'audio-index-near-shuffled.jsonl.gz'
    with gzip.open(idx_path, 'wt') as f:
        for row in index_rows:
            f.write(json.dumps(row) + '\n')
    entries = yaml.safe_load((cat / 'catalog.yaml').read_text())
    e = entries[0]
    e['artifacts'] = [a for a in e['artifacts'] if a['name'] not in (f'{split}-records', 'audio-index-near-shuffled')]
    e['artifacts'] += [
        {'name': 'audio-index-near-shuffled', 'kind': 'audio-index', 'root_alias': 'meeting_simreal_probe_catalog',
         'relative_path': idx_path.name, 'sha256': sha256_file(idx_path), 'metadata': {'record_count': len(index_rows)}},
        {'name': f'{split}-records', 'kind': 'audio-records', 'root_alias': 'meeting_simreal_probe_catalog',
         'relative_path': f'records/{split}.jsonl.gz', 'sha256': sha256_file(rec_path),
         'metadata': {'record_count': len(out_records)}}]
    e['splits'][split] = {'artifacts': {'records': [f'{split}-records'], 'audio_index': ['audio-index-near-shuffled']},
                          'statistics': {'records': len(out_records)}}
    (cat / 'catalog.yaml').write_text(yaml.safe_dump(entries, sort_keys=False, allow_unicode=True))
    print('done', len(out_records), flush=True)


if __name__ == '__main__':
    main()
