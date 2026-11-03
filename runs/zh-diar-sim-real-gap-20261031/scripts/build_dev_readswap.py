"""AliMeeting dev windows with the real turn timing but read speech in place of the spontaneous turns (style probe).

Every reference turn of more than 2 units is replaced by an AISHELL-1 dev/test utterance (speakers unseen in
training; one AISHELL speaker per meeting speaker) whose trimmed length is closest to the turn's but still ends before
the same speaker's next turn (so one speaker never overlaps itself), placed at the turn's start at the level of the
real turn in its headset. Backchannels (<= 2 units) and turns no utterance fits keep their real headset audio and text;
those are marked ``kept_real`` in the record's turns and left out of the comparison. Arrangement (who overlaps whom, when, how much) stays real; only the speech style changes. Compared
with near_gated and near_retimed per overlap category this separates speech style from overlap arrangement.
Record ids "read_realtime/<panel id>"."""
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

PAD, DBFS, HOP = 0.1, -26.0, 160


def trim(x):
    n = len(x) // HOP
    e = 10 * np.log10((x[:n * HOP].reshape(n, HOP) ** 2).mean(1) + 1e-10)
    on = np.where(e > e.max() - 35)[0]
    return x[on[0] * HOP:(on[-1] + 1) * HOP] if len(on) else x


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--panel-records', type=Path, required=True)
    p.add_argument('--near-dir', type=Path, required=True)
    p.add_argument('--aishell', type=Path, required=True, help='aishell legacy-20260804 version dir')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--catalog-dir', type=Path, required=True)
    args = p.parse_args()
    split = 'read_realtime'
    (args.output / split).mkdir(parents=True, exist_ok=True)
    pool = defaultdict(list)
    for part in ('dev', 'test'):
        recs = {r['id']: r for r in map(json.loads, gzip.open(args.aishell / f'{part}_rec_0/aishell_recordings_{part}.jsonl.gz', 'rt'))}
        for s in map(json.loads, gzip.open(args.aishell / f'{part}_sup_0/aishell_supervisions_{part}.jsonl.gz', 'rt')):
            pool[s['speaker']].append((recs[s['recording_id']]['sources'][0]['source'], s['duration'], s['text']))
    aishell_speakers = sorted(pool)
    records = [json.loads(line) for line in gzip.open(args.panel_records, 'rt')]
    by_meeting = defaultdict(list)
    for r in records:
        by_meeting[r['metadata']['meeting_id']].append(r)
    out_records, index_rows, mismatch = [], [], []
    for meeting, rows in sorted(by_meeting.items()):
        spans = defaultdict(list)
        for r in rows:
            s0 = r['metadata']['source_record']['start']
            for t in r['metadata']['turns']:
                spans[t['source_speaker']].append((s0 + t['start'], s0 + t['end']))
        rng = np.random.default_rng(stable_int(20261031, meeting))
        mapping = dict(zip(sorted(spans), rng.choice(aishell_speakers, size=len(spans), replace=False)))
        trimmed = {}
        for aspk in mapping.values():
            for path, _, text in pool[aspk]:
                trimmed[path] = trim(load_channel0(path))
        tracks, used = {}, set()
        for spk in spans:
            (path,) = glob.glob(f'{args.near_dir}/{meeting}_N_SPK{spk}.wav')
            x = load_channel0(path)
            tracks[spk] = x * activity(spans[spk], len(x), 0.25, 0)
        for r in rows:
            s0 = r['metadata']['source_record']['start']
            frames = round(r['audio_slots'][0]['ref']['duration'] * RATE)
            audio = np.zeros(frames + 20 * RATE, np.float32)
            new_turns = []
            ordered = sorted(r['metadata']['turns'], key=lambda t: t['start'])
            for i, t in enumerate(ordered):
                spk, d = t['source_speaker'], t['end'] - t['start']
                nxt = [u['start'] for u in ordered[i + 1:] if u['source_speaker'] == spk]
                slot = (nxt[0] - t['start'] - 0.05) if nxt else d + 3.0
                real = tracks[spk][max(0, round((s0 + t['start']) * RATE)):round((s0 + t['end']) * RATE)]
                level = np.sqrt(np.mean(real ** 2) + 1e-12)
                cands = [c for c in pool[mapping[spk]] if c[0] not in used and len(trimmed[c[0]]) / RATE <= slot]
                kept = len(t['text']) <= 2 or not cands
                if kept:
                    seg, text, dur = real, t['text'], d
                else:
                    path, _, text = min(cands, key=lambda c: abs(len(trimmed[c[0]]) / RATE - d))
                    used.add(path)
                    seg = trimmed[path] * (level / (np.sqrt(np.mean(trimmed[path] ** 2)) + 1e-12))
                    dur = len(seg) / RATE
                    mismatch.append(dur - d)
                at = round(t['start'] * RATE)
                audio[at:at + len(seg)] += seg[:len(audio) - at]
                new_turns.append({**t, 'text': text, 'start': round(t['start'], 2), 'end': round(t['start'] + dur, 2),
                                  'kept_real': kept})
            end = max([frames] + [round(u['end'] * RATE) for u in new_turns])
            audio = audio[:end]
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
        print(meeting, mapping, flush=True)
    m = np.array(mismatch)
    print(f'duration change (read - real) median {np.median(m):+.2f} s, |p90| {np.percentile(np.abs(m), 90):.2f} s', flush=True)
    cat = args.catalog_dir
    rec_path = cat / 'records' / f'{split}.jsonl.gz'
    with gzip.open(rec_path, 'wt') as f:
        for row in out_records:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    idx_path = cat / 'audio-index-read-realtime.jsonl.gz'
    with gzip.open(idx_path, 'wt') as f:
        for row in index_rows:
            f.write(json.dumps(row) + '\n')
    entries = yaml.safe_load((cat / 'catalog.yaml').read_text())
    e = entries[0]
    e['artifacts'] = [a for a in e['artifacts'] if a['name'] not in (f'{split}-records', 'audio-index-read-realtime')]
    e['artifacts'] += [
        {'name': 'audio-index-read-realtime', 'kind': 'audio-index', 'root_alias': 'meeting_simreal_probe_catalog',
         'relative_path': idx_path.name, 'sha256': sha256_file(idx_path), 'metadata': {'record_count': len(index_rows)}},
        {'name': f'{split}-records', 'kind': 'audio-records', 'root_alias': 'meeting_simreal_probe_catalog',
         'relative_path': f'records/{split}.jsonl.gz', 'sha256': sha256_file(rec_path),
         'metadata': {'record_count': len(out_records)}}]
    e['splits'][split] = {'artifacts': {'records': [f'{split}-records'], 'audio_index': ['audio-index-read-realtime']},
                          'statistics': {'records': len(out_records)}}
    (cat / 'catalog.yaml').write_text(yaml.safe_dump(entries, sort_keys=False, allow_unicode=True))
    print('done', len(out_records), flush=True)


if __name__ == '__main__':
    main()
