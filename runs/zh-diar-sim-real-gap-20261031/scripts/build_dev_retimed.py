"""AliMeeting dev windows whose real turns are re-arranged the way our synthesizers arrange overlap (arrangement probe).

Each turn is cut from its own gated headset ([start - pad, end + pad]) as in build_dev_sequential.py, then laid out in
the original order: at a speaker change the next turn starts early so that it overlaps the previous one by a
fraction U(0.2, 0.9) of the shorter of the two (probability 0.6; sentence-on-sentence overlap at turn boundaries, as in
the read-speech mixtures and the 10-13 Emilia2 re-timing), and backchannels (<= 2 units) are dropped at a random point
inside the previous other-speaker turn (probability 0.6). Same speech, same channel, same texts and speakers; only the
overlap arrangement is synthetic. Compared with near_gated per overlap category (computed on each layout's own
references). Record ids "near_retimed/<panel id>"."""
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

from amphiondata.multispeaker.farfield import activity, load_channel0, stable_int
from amphiondata.multispeaker.meetings import RATE

PAD, GAP, DBFS = 0.15, 0.3, -26.0
P_OVERLAP, P_INSIDE = 0.6, 0.6


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--panel-records', type=Path, required=True)
    p.add_argument('--near-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--catalog-dir', type=Path, required=True)
    args = p.parse_args()
    split = 'near_retimed'
    (args.output / split).mkdir(parents=True, exist_ok=True)
    records = [json.loads(line) for line in gzip.open(args.panel_records, 'rt')]
    by_meeting = defaultdict(list)
    for r in records:
        by_meeting[r['metadata']['meeting_id']].append(r)
    out_records, index_rows = [], []
    for meeting, rows in sorted(by_meeting.items()):
        spans = defaultdict(list)
        for r in rows:
            s0 = r['metadata']['source_record']['start']
            for t in r['metadata']['turns']:
                spans[t['source_speaker']].append((s0 + t['start'], s0 + t['end']))
        tracks = {}
        for spk in spans:
            (path,) = glob.glob(f'{args.near_dir}/{meeting}_N_SPK{spk}.wav')
            x = load_channel0(path)
            tracks[spk] = x * activity(spans[spk], len(x), 0.25, 0)
        for r in rows:
            rng = np.random.default_rng(stable_int(20261031, r['id']))
            s0 = r['metadata']['source_record']['start']
            placed, cursor, prev = [], 0.5, None
            for t in sorted(r['metadata']['turns'], key=lambda t: t['start']):
                x = tracks[t['source_speaker']]
                seg = x[max(0, round((s0 + t['start'] - PAD) * RATE)):round((s0 + t['end'] + PAD) * RATE)]
                d = t['end'] - t['start']
                start = cursor + PAD
                if prev is not None and prev[0]['source_speaker'] != t['source_speaker']:
                    pt, ps, pe = prev
                    if len(t['text']) <= 2 and pe - ps > 1.0 and rng.random() < P_INSIDE:
                        start = ps + rng.uniform(0.2, 0.8) * (pe - ps)
                    elif rng.random() < P_OVERLAP:
                        start = max(ps, pe - rng.uniform(0.2, 0.9) * min(d, pe - ps))
                placed.append((t, start, seg))
                end = start + d
                cursor = max(cursor, end + PAD + GAP)
                if len(t['text']) > 2 or prev is None:
                    prev = (t, start, end)
            total = max(round((s + len(seg) / RATE) * RATE) for _, s, seg in placed) + RATE // 2
            audio = np.zeros(total, np.float32)
            new_turns = []
            for t, s, seg in placed:
                at = max(0, round((s - PAD) * RATE))
                audio[at:at + len(seg)] += seg[:total - at]
                new_turns.append({**t, 'start': round(s, 2), 'end': round(s + t['end'] - t['start'], 2)})
            new_turns.sort(key=lambda u: u['start'])
            act = activity([(u['start'], u['end']) for u in new_turns], len(audio), 0, 0) > 0
            audio = np.clip(audio * (10 ** (DBFS / 20) / np.sqrt(np.mean(audio[act] ** 2) + 1e-12)), -0.99, 0.99)
            rid = r['id']
            sf.write(args.output / split / f'{rid}.wav', audio, RATE, subtype='PCM_16')
            row = json.loads(json.dumps(r))
            row['id'] = f'{split}/{rid}'
            row['audio_slots'][0]['ref'].update({'dataset_id': 'meeting_simreal_probe', 'version': 'v1-20261031',
                                                 'split': split, 'cut_id': f'{split}/{rid}', 'start': 0.0,
                                                 'duration': len(audio) / RATE})
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
    idx_path = cat / 'audio-index-near-retimed.jsonl.gz'
    with gzip.open(idx_path, 'wt') as f:
        for row in index_rows:
            f.write(json.dumps(row) + '\n')
    entries = yaml.safe_load((cat / 'catalog.yaml').read_text())
    e = entries[0]
    e['artifacts'] = [a for a in e['artifacts'] if a['name'] not in (f'{split}-records', 'audio-index-near-retimed')]
    e['artifacts'] += [
        {'name': 'audio-index-near-retimed', 'kind': 'audio-index', 'root_alias': 'meeting_simreal_probe_catalog',
         'relative_path': idx_path.name, 'sha256': sha256_file(idx_path), 'metadata': {'record_count': len(index_rows)}},
        {'name': f'{split}-records', 'kind': 'audio-records', 'root_alias': 'meeting_simreal_probe_catalog',
         'relative_path': f'records/{split}.jsonl.gz', 'sha256': sha256_file(rec_path),
         'metadata': {'record_count': len(out_records)}}]
    e['splits'][split] = {'artifacts': {'records': [f'{split}-records'], 'audio_index': ['audio-index-near-retimed']},
                          'statistics': {'records': len(out_records)}}
    (cat / 'catalog.yaml').write_text(yaml.safe_dump(entries, sort_keys=False, allow_unicode=True))
    print('done', len(out_records), flush=True)


if __name__ == '__main__':
    main()
