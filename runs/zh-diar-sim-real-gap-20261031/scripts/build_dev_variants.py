"""Render the 138 AliMeeting dev selection-panel windows under controlled acoustic conditions (sim-vs-real probe).

Every variant keeps the panel's windows, references and record ids (meeting_selection_panel@v1-20261008:
alimeeting_dev), so each is scored window-by-window against the real far-field panel. Near-to-far rendering follows
the training builder (amphiondata.multispeaker.farfield, used for alimeeting_near_farsim_meeting_sot@near-farsim-v1-20261009):
same SLR28 real-RIR rooms (>= 4 positions), MUSAN noise, SNR 0-20 dB, +-3 dB levels, 1 s RIRs, -26 dBFS, headsets gated
to their own annotated turns. ``render`` replicates render_far_field draw for draw (checked against it on the first
meeting) and only swaps one input at a time:

  far_mean        real far field, mean of the 8 array channels (= the panel audio; checked by correlation)
  far_ch0         real far field, channel 0 only
  near_gated      headsets gated to their own turns, summed, dry
  near_raw        headsets summed as recorded (cross-talk leakage and unannotated sounds kept), dry
  sim_train       the training simulation
  sim_quiet       sim_train without noise (same room, positions, levels)
  sim_reallevel   sim_train with each speaker's level measured in the real far field instead of +-3 dB
  sim_roomnoise   sim_train with this meeting's own far-field room noise at its measured SNR instead of MUSAN
  sim_real        both: measured levels and the room's own noise at its measured SNR
"""
import argparse
import glob
import gzip
import hashlib
import json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import oaconvolve

from amphiondata.multispeaker.farfield import activity, envelope_lag, load_channel0, render_far_field, room_of, stable_int
from amphiondata.multispeaker.meetings import RATE, read_rows

VARIANTS = ('far_mean', 'far_ch0', 'near_gated', 'near_raw', 'sim_train', 'sim_quiet', 'sim_reallevel',
            'sim_roomnoise', 'sim_real')
# Training builder parameters (report.json of near-farsim-v1-20261009).
TRAIN = {'seed': 20261009, 'snr_db': [0.0, 20.0], 'level_db': 3.0, 'pad': 0.25, 'ramp': 0.02, 'rir_seconds': 1.0,
         'speech_dbfs': -26.0}
# Room-noise estimate: 0.5 s chunks at least GUARD s away from any annotated turn; chunks more than 3 dB above the
# median chunk are dropped (unannotated talk or knocks), the rest is the room's noise.
CHUNK, GUARD = 0.5, 0.5


def scale_to(signal, active, dbfs):
    power = np.mean(signal[active] ** 2) if active.any() else np.mean(signal ** 2)
    return signal * (10 ** (dbfs / 20) / np.sqrt(power + 1e-12))


def room_noise(far, turns, frames):
    busy = activity(turns, frames, GUARD, 0) > 0
    hop = round(CHUNK * RATE)
    chunks = [far[i:i + hop] for i in range(0, frames - hop, hop) if not busy[i:i + hop].any()]
    power = np.array([np.mean(c ** 2) for c in chunks])
    keep = [c for c, p in zip(chunks, power) if p <= 2 * np.median(power)]
    return np.concatenate(keep), float(np.mean([np.mean(c ** 2) for c in keep])), len(keep), len(chunks)


def render(streams, level_turns, turns, frames, spec, rng, gains=None, noise=None, snr=None):
    """render_far_field with optional overrides; the random draws happen in the same order either way."""
    rooms = spec['rooms']
    room = sorted(rooms)[rng.integers(len(rooms))]
    positions = list(rng.permutation(len(rooms[room])))
    wet, used = [], {}
    for index, (name, dry) in enumerate(streams):
        rir_id, rir_path = rooms[room][positions[index % len(positions)]]
        rir = load_channel0(rir_path)
        rir = rir[int(np.abs(rir).argmax()):][: round(spec['rir_seconds'] * RATE)]
        rir = rir / np.linalg.norm(rir)
        signal = oaconvolve(dry, rir)[:frames].astype(np.float32)
        active = activity(level_turns[name], frames, 0, 0) > 0
        level = np.sqrt(np.mean(signal[active] ** 2)) + 1e-9
        gain_db = float(rng.uniform(-spec['level_db'], spec['level_db']))
        if gains is not None:
            gain_db = gains[name]
        signal *= 10 ** (gain_db / 20) / level
        used[name] = {'rir': rir_id, 'gain_db': round(gain_db, 2)}
        wet.append(signal)
    speech = np.sum(wet, axis=0)
    del wet
    active = activity(turns, frames, 0, 0) > 0
    speech_power = np.mean(speech[active] ** 2) if active.any() else np.mean(speech ** 2)
    if noise is None:
        noise_signal = np.zeros(frames, dtype=np.float32)
        filled = 0
        fade = np.linspace(0, 1, round(0.01 * RATE), dtype=np.float32)
        while filled < frames:
            _, clip_path = spec['noises'][rng.integers(len(spec['noises']))]
            clip = load_channel0(clip_path)
            rms = np.sqrt(np.mean(clip ** 2))
            if rms < 1e-5 or len(clip) < 2 * len(fade):
                continue
            clip = clip / rms
            clip[: len(fade)] *= fade
            clip[-len(fade):] *= fade[::-1]
            take = min(len(clip), frames - filled)
            noise_signal[filled: filled + take] = clip[:take]
            filled += take
        drawn = float(rng.uniform(*spec['snr_db']))
    else:
        reps = int(np.ceil(frames / len(noise)))
        noise_signal = np.tile(noise, reps)[:frames]
        noise_signal = noise_signal / np.sqrt(np.mean(noise_signal ** 2))
        drawn = None
    snr_db = drawn if snr is None else snr
    mixture = speech + noise_signal * np.sqrt(speech_power / 10 ** (snr_db / 10))
    scale = 10 ** (spec['speech_dbfs'] / 20) / np.sqrt(speech_power)
    peak = np.abs(mixture).max() * scale
    if peak > 0.99:
        scale *= 0.99 / peak
    return (mixture * scale).astype(np.float32), {'room': room, 'positions': used, 'snr_db': round(snr_db, 2)}


def meeting_job(job):
    meeting, frames_hint, spec, out, check = job['meeting'], job['frames'], job['spec'], Path(job['out']), job['check']
    far_path = glob.glob(f"{spec['far_dir']}/{meeting}_MS*.wav")
    assert len(far_path) == 1, (meeting, far_path)
    far8, rate = sf.read(far_path[0], dtype='float32', always_2d=True)
    assert rate == RATE
    frames = far8.shape[0]
    by_speaker = defaultdict(list)
    for speaker, start, end in job['turns']:
        by_speaker[speaker].append((start, end))
    speakers = sorted(by_speaker)
    turns = [t for rows in by_speaker.values() for t in rows]
    active = activity(turns, frames, 0, 0) > 0
    far_mean = far8.mean(axis=1)
    far_ch0 = np.ascontiguousarray(far8[:, 0])
    del far8
    stats = {'meeting': meeting, 'speakers': len(speakers), 'frames': frames}

    near, gated = {}, {}
    for speaker in speakers:
        (path,) = glob.glob(f"{spec['near_dir']}/{meeting}_N_SPK{speaker}.wav")
        track = load_channel0(path)
        track = np.pad(track[:frames], (0, max(0, frames - len(track))))
        lag, peak, at_zero = envelope_lag(track, by_speaker[speaker])
        stats.setdefault('near_lag_s', {})[speaker] = lag
        near[speaker] = track
        gated[speaker] = track * activity(by_speaker[speaker], frames, spec['pad'], spec['ramp'])
    stats['far_lag_s'] = envelope_lag(far_mean, turns)[0]

    # Real per-speaker level in the far field: power on the speaker's exclusive (no one else annotated) turns,
    # minus the room-noise power; centred over the meeting's speakers.
    noise, noise_power, kept, chunks = room_noise(far_mean, turns, frames)
    masks = {s: activity(by_speaker[s], frames, 0, 0) > 0 for s in speakers}
    count = np.sum([m.astype(np.int8) for m in masks.values()], axis=0)
    level = {}
    for s in speakers:
        excl = masks[s] & (count == 1)
        p = np.mean(far_mean[excl] ** 2) - noise_power if excl.sum() > RATE else np.nan
        level[s] = 10 * np.log10(max(p, 1e-12))
    mean_level = np.nanmean(list(level.values()))
    real_gains = {s: (float(level[s] - mean_level) if np.isfinite(level[s]) else 0.0) for s in speakers}
    speech_power = np.mean(far_mean[active] ** 2) - noise_power
    real_snr = float(10 * np.log10(max(speech_power, 1e-12) / noise_power))
    stats.update(real_gains_db=real_gains, real_snr_db=real_snr, room_noise_chunks=[kept, chunks])

    streams = [(s, gated[s]) for s in speakers]
    rng = lambda: np.random.default_rng(stable_int(spec['seed'], meeting))  # noqa: E731
    audio = {
        'far_mean': far_mean, 'far_ch0': far_ch0,
        'near_gated': scale_to(np.sum([gated[s] for s in speakers], axis=0), active, spec['speech_dbfs']),
        'near_raw': scale_to(np.sum([near[s] for s in speakers], axis=0), active, spec['speech_dbfs']),
    }
    audio['sim_train'], stats['sim_train'] = render(streams, by_speaker, turns, frames, spec, rng())
    if check:
        reference, _ = render_far_field(streams, by_speaker, turns, frames, spec, rng())
        stats['replica_max_abs_diff'] = float(np.abs(reference - audio['sim_train']).max())
        assert stats['replica_max_abs_diff'] < 1e-6, stats['replica_max_abs_diff']
    audio['sim_quiet'], stats['sim_quiet'] = render(streams, by_speaker, turns, frames, spec, rng(), snr=100.0)
    audio['sim_reallevel'], stats['sim_reallevel'] = render(streams, by_speaker, turns, frames, spec, rng(),
                                                            gains=real_gains)
    audio['sim_roomnoise'], stats['sim_roomnoise'] = render(streams, by_speaker, turns, frames, spec, rng(),
                                                            noise=noise, snr=real_snr)
    audio['sim_real'], stats['sim_real'] = render(streams, by_speaker, turns, frames, spec, rng(), gains=real_gains,
                                                  noise=noise, snr=real_snr)
    written = []
    for record in job['records']:
        start = round(record['start'] * RATE)
        n = round(record['duration'] * RATE)
        for variant in VARIANTS:
            piece = audio[variant][start:start + n]
            path = out / variant / f"{record['id']}.wav"
            sf.write(path, piece, RATE, subtype='PCM_16')
            written.append((variant, record['id'], len(piece)))
    return stats, written


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--panel-records', type=Path, required=True)
    p.add_argument('--panel-audio-dir', type=Path, required=True)
    p.add_argument('--far-dir', type=Path, required=True)
    p.add_argument('--near-dir', type=Path, required=True)
    p.add_argument('--rir-manifest', type=Path, required=True)
    p.add_argument('--noise-manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--catalog-dir', type=Path, required=True)
    p.add_argument('--version', default='v1-20261031')
    p.add_argument('--workers', type=int, default=8)
    args = p.parse_args()

    rooms = defaultdict(list)
    for row in read_rows(args.rir_manifest):
        room = room_of(row['id'])
        if room:
            rooms[room].append((row['id'], row['sources'][0]['source']))
    rooms = {k: sorted(v) for k, v in rooms.items() if len(v) >= 4}
    noises = sorted((r['id'], r['sources'][0]['source']) for r in read_rows(args.noise_manifest))
    spec = {**TRAIN, 'rooms': rooms, 'noises': noises, 'far_dir': str(args.far_dir), 'near_dir': str(args.near_dir)}

    with gzip.open(args.panel_records, 'rt') as f:
        records = [json.loads(line) for line in f]
    by_meeting = defaultdict(list)
    turns = defaultdict(dict)
    for r in records:
        meta = r['metadata']
        start = meta['source_record']['start']
        by_meeting[meta['meeting_id']].append({'id': r['id'], 'start': start,
                                               'duration': r['audio_slots'][0]['ref']['duration']})
        for t in meta['turns']:
            turns[meta['meeting_id']][t['source_id']] = (t['source_speaker'], start + t['start'], start + t['end'])
    for variant in VARIANTS:
        (args.output / variant).mkdir(parents=True, exist_ok=True)
    jobs = [{'meeting': m, 'frames': None, 'spec': spec, 'out': str(args.output), 'records': rows,
             'turns': sorted(turns[m].values(), key=lambda t: t[1]), 'check': i == 0}
            for i, (m, rows) in enumerate(sorted(by_meeting.items()))]
    stats, written = [], []
    with ProcessPoolExecutor(args.workers) as pool:
        for s, w in pool.map(meeting_job, jobs):
            stats.append(s)
            written += w
            print(json.dumps({k: v for k, v in s.items() if k in ('meeting', 'real_snr_db', 'real_gains_db',
                                                                   'replica_max_abs_diff')}), flush=True)

    # The real far-field mean must reproduce the panel audio.
    check = []
    for r in records[:: max(1, len(records) // 10)]:
        ours, _ = sf.read(args.output / 'far_mean' / f"{r['id']}.wav", dtype='float32')
        panel, _ = sf.read(args.panel_audio_dir / f"{r['id']}.wav", dtype='float32')
        n = min(len(ours), len(panel))
        check.append(float(np.corrcoef(ours[:n], panel[:n])[0, 1]))
    print('far_mean vs panel audio correlation:', [round(c, 4) for c in check], flush=True)

    # Catalog: one split per variant, records identical to the panel's apart from the audio reference.
    cat = args.catalog_dir
    (cat / 'records').mkdir(parents=True, exist_ok=True)
    index_rows = []
    for variant, rid, n in written:
        index_rows.append({'channels': 1, 'cut_id': f'{variant}/{rid}', 'duration': n / RATE, 'num_frames': n,
                           'relative_path': f'{variant}/{rid}.wav', 'root_alias': 'meeting_simreal_probe',
                           'sample_rate': RATE})
    index_path = cat / 'audio-index.jsonl.gz'
    with gzip.open(index_path, 'wt') as f:
        for row in index_rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    artifacts = [{'name': 'audio-index', 'kind': 'audio-index', 'root_alias': 'meeting_simreal_probe_catalog',
                  'relative_path': 'audio-index.jsonl.gz', 'sha256': sha256_file(index_path),
                  'metadata': {'record_count': len(index_rows)}}]
    splits = {}
    for variant in VARIANTS:
        path = cat / 'records' / f'{variant}.jsonl.gz'
        with gzip.open(path, 'wt') as f:
            for r in records:
                row = json.loads(json.dumps(r))
                row['audio_slots'][0]['ref'].update({'dataset_id': 'meeting_simreal_probe', 'version': args.version,
                                                     'split': variant, 'cut_id': f"{variant}/{r['id']}"})
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
        artifacts.append({'name': f'{variant}-records', 'kind': 'audio-records',
                          'root_alias': 'meeting_simreal_probe_catalog', 'relative_path': f'records/{variant}.jsonl.gz',
                          'sha256': sha256_file(path), 'metadata': {'record_count': len(records)}})
        splits[variant] = {'artifacts': {'records': [f'{variant}-records'], 'audio_index': ['audio-index']},
                           'statistics': {'records': len(records)}}
    entry = {'schema_version': 'dataset-catalog/2.0', 'dataset_id': 'meeting_simreal_probe', 'version': args.version,
             'languages': ['zh'], 'tasks': ['speaker_attributed_asr'], 'artifacts': artifacts, 'splits': splits,
             'provenance': {'status': 'complete', 'synthetic': 'partial',
                            'description': 'AliMeeting 官方 Eval 8 场 138 个选模窗口的受控声学变体（见 scripts/build_dev_variants.py），只用于分析，不进训练。'}}
    import yaml
    (cat / 'catalog.yaml').write_text(yaml.safe_dump([entry], sort_keys=False, allow_unicode=True))
    (cat / 'roots.json').write_text(json.dumps({'meeting_simreal_probe': str(args.output.resolve()),
                                                'meeting_simreal_probe_catalog': str(cat.resolve())}, indent=2) + '\n')
    (cat / 'variant-stats.json').write_text(json.dumps(stats, indent=2, ensure_ascii=False) + '\n')
    print('done', len(written), 'files', flush=True)


if __name__ == '__main__':
    main()
