"""Build the 180 s meeting benchmark: deployment-length windows of real meetings plus
far-field synthetic meetings with 6-10 speakers.

Real meetings are cut near the target length at a gap between utterances, so no
utterance is split; when no gap exists the crossing utterances are kept whole and the
window grows to include them. Every utterance in a window is in its reference.
Synthetic meetings use held-out test-split speakers, real RIRs per speaker and noise.
"""
import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml
from lhotse import RecordingSet, SupervisionSet
from scipy.signal import fftconvolve

RATE = 16000


def write_clip(output, clip_id, audio):
    path = output / 'audio' / f'{clip_id}.wav'
    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    if peak > 0.99:
        audio = audio * (0.99 / peak)
    sf.write(str(path), audio, RATE, subtype='PCM_16')
    return str(path), hashlib.sha256(path.read_bytes()).hexdigest()


def overlap_seconds(segments, duration, step=0.01):
    active = np.zeros(int(duration / step) + 2, dtype=np.int16)
    for speaker in {s['speaker'] for s in segments}:
        mask = np.zeros_like(active, dtype=bool)
        for s in segments:
            if s['speaker'] == speaker:
                mask[int(max(s['start'], 0) / step):int(min(s['end'], duration) / step)] = True
        active += mask
    return float((active >= 2).sum() * step)


def windows(utterances, total, target, low, high, limit):
    """Yield (start, end) windows that never cut through an utterance."""
    utterances = sorted(utterances, key=lambda u: u['start'])
    start = max(0.0, utterances[0]['start'] - 0.5)
    fallbacks = 0
    while start < total:
        def busy(t):
            return any(u['start'] < t < u['end'] for u in utterances)
        candidates = [t for t in np.arange(start + low, min(start + high, total), 0.1) if not busy(t)]
        if candidates:
            end = min(candidates, key=lambda t: abs(t - start - target))
        else:
            end = min(start + target, total)
            while busy(end) and end < min(start + limit, total):
                end = max(u['end'] for u in utterances if u['start'] < end < u['end']) + 0.05
            fallbacks += 1
        end = min(end, total)
        yield start, end, fallbacks
        start = end


def real_meetings(spec, output, rng, settings):
    recordings = RecordingSet.from_file(spec['recordings'])
    supervisions = SupervisionSet.from_file(spec['supervisions'])
    by_recording = {}
    for s in supervisions:
        if s.text and s.text.strip():
            by_recording.setdefault(s.recording_id, []).append(s)
    clips, fallbacks = [], 0
    for recording in sorted(recordings, key=lambda r: r.id):
        utterances = [{'speaker': s.speaker, 'start': s.start, 'end': s.end, 'text': s.text.strip()}
                      for s in by_recording.get(recording.id, [])]
        if not utterances:
            continue
        cut = list(windows(utterances, recording.duration, settings['target_seconds'],
                           settings['min_seconds'], settings['max_seconds'], settings['limit_seconds']))
        fallbacks += cut[-1][2] if cut else 0
        cut = [(a, b) for a, b, _ in cut if b - a >= settings['min_seconds']
               and any(a <= u['start'] and u['end'] <= b for u in utterances)]
        if len(cut) > spec['windows_per_meeting']:
            picks = np.linspace(0, len(cut) - 1, spec['windows_per_meeting']).round().astype(int)
            cut = [cut[i] for i in sorted(set(picks))]
        for index, (a, b) in enumerate(cut):
            audio = recording.load_audio(offset=a, duration=b - a)
            audio = np.asarray(audio, dtype=np.float32).mean(axis=0)
            reference = [{'speaker': u['speaker'], 'start': round(u['start'] - a, 3),
                          'end': round(u['end'] - a, 3), 'text': u['text']}
                         for u in utterances if a <= u['start'] and u['end'] <= b]
            # Original speaker IDs become first-speech-ordered labels, as in training targets.
            order = {}
            for r in sorted(reference, key=lambda r: r['start']):
                order.setdefault(r['speaker'], f'S{len(order) + 1}')
            for r in reference:
                r['speaker'] = order[r['speaker']]
            clip_id = f"{spec['label']}-{recording.id}-{index:02d}"
            path, digest = write_clip(output, clip_id, audio)
            duration = round(len(audio) / RATE, 3)
            clips.append({'id': clip_id, 'dataset': spec['label'], 'language': spec['language'],
                          'kind': 'real', 'meeting_id': f"{spec['label']}:{recording.id}",
                          'duration': duration, 'reference': reference,
                          'reference_speakers': len(order),
                          'overlap_seconds': overlap_seconds(reference, duration),
                          'source_start': round(a, 3), 'source_audio': recording.id,
                          'channel_policy': 'mean of all channels', 'audio': path,
                          'audio_sha256': digest})
    return clips, fallbacks


def utterance_pool(spec, max_seconds):
    recordings = {r.id: r for r in RecordingSet.from_file(spec['recordings'])}
    pool, missing = {}, 0
    for s in SupervisionSet.from_file(spec['supervisions']):
        if s.recording_id not in recordings:
            missing += 1
            continue
        if s.text and s.duration <= max_seconds and s.duration >= 1.0:
            pool.setdefault(s.speaker, []).append((s, recordings[s.recording_id]))
    if missing:
        print(json.dumps({'supervisions_without_recording': missing, 'source': str(spec['supervisions'])}))
    return pool


def synthetic_meetings(settings, output, rng):
    syn = settings['synthetic']
    pools = {lang: utterance_pool(spec, syn['max_utterance_seconds']) for lang, spec in syn['sources'].items()}
    rirs = list(RecordingSet.from_file(syn['rir_recordings']))
    noises = [r for r in RecordingSet.from_file(syn['noise_recordings'])
              if syn['noise_subset'] in r.sources[0].source]
    clips = []
    for language, mix in syn['languages'].items():
        for count in syn['speaker_counts']:
            for sample in range(syn['samples_per_cell']):
                speakers = []
                for lang, share in mix.items():
                    speakers += [(lang, s) for s in rng.sample(sorted(pools[lang]), round(count * share))]
                rng.shuffle(speakers)
                order, timeline, cursor, previous = [], [], 0.0, None
                remaining = {k: rng.sample(pools[k[0]][k[1]], len(pools[k[0]][k[1]])) for k in speakers}
                while cursor < syn['target_seconds'] - 5:
                    choices = [k for k in speakers if k not in order] or [k for k in speakers if k != previous]
                    key = rng.choice(choices)
                    if not remaining[key]:
                        speakers.remove(key)
                        continue
                    supervision, recording = remaining[key].pop()
                    gap = rng.uniform(*syn['gap_seconds'])
                    if timeline and rng.random() < syn['overlap_probability']:
                        gap = -min(rng.uniform(*syn['overlap_seconds']), timeline[-1]['end'] - timeline[-1]['start'] - 0.3)
                    start = max(0.0, cursor + gap)
                    timeline.append({'key': key, 'start': start, 'end': start + supervision.duration,
                                     'supervision': supervision, 'recording': recording})
                    order.append(key) if key not in order else None
                    cursor, previous = max(cursor, start + supervision.duration), key
                duration = cursor + 0.5
                audio = np.zeros(int(duration * RATE) + RATE, dtype=np.float32)
                speaker_rir = {k: rirs[rng.randrange(len(rirs))] for k in order}
                reference = []
                labels = {k: f'S{i + 1}' for i, k in enumerate(order)}
                for item in timeline:
                    s = item['supervision']
                    wave = np.asarray(item['recording'].load_audio(offset=s.start, duration=s.duration),
                                      dtype=np.float32)[0]
                    wave = wave / (np.sqrt(np.mean(wave ** 2)) + 1e-8) * 0.05 * 10 ** (rng.uniform(-3, 3) / 20)
                    rir = np.asarray(speaker_rir[item['key']].load_audio(), dtype=np.float32)[0]
                    rir = rir[np.argmax(np.abs(rir)):]  # direct path at t=0 keeps reference times
                    wave = fftconvolve(wave, rir / (np.linalg.norm(rir) + 1e-8))[:len(wave) + RATE // 2]
                    begin = int(item['start'] * RATE)
                    audio[begin:begin + len(wave)] += wave[:len(audio) - begin]
                    reference.append({'speaker': labels[item['key']], 'start': round(item['start'], 3),
                                      'end': round(item['end'], 3), 'text': s.text.strip()})
                audio = audio[:int(duration * RATE)]
                noise = np.asarray(noises[rng.randrange(len(noises))].load_audio(), dtype=np.float32).mean(axis=0)
                noise = np.resize(noise, len(audio))
                snr = rng.uniform(*syn['snr_db'])
                noise *= np.sqrt(np.mean(audio ** 2) / (np.mean(noise ** 2) + 1e-12) / 10 ** (snr / 10))
                audio = audio + noise
                clip_id = f'synthetic-{language}-{count}spk-{sample:02d}'
                path, digest = write_clip(output, clip_id, audio)
                clips.append({'id': clip_id, 'dataset': f'synthetic_{language}', 'language': language,
                              'kind': 'synthetic', 'meeting_id': clip_id, 'duration': round(duration, 3),
                              'reference': sorted(reference, key=lambda r: r['start']),
                              'reference_speakers': len(order),
                              'overlap_seconds': overlap_seconds(reference, duration),
                              'snr_db': round(snr, 2), 'audio': path, 'audio_sha256': digest})
    return clips


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    settings = yaml.safe_load(parser.parse_args().config.read_text())['parameters']
    output = Path(settings['output'])
    (output / 'audio').mkdir(parents=True, exist_ok=True)
    rng = random.Random(settings['seed'])
    clips, report = [], {}
    for spec in settings['real']:
        rows, fallbacks = real_meetings(spec, output, rng, settings)
        clips += rows
        report[spec['label']] = {'clips': len(rows), 'meetings': len({r['meeting_id'] for r in rows}),
                                 'hours': round(sum(r['duration'] for r in rows) / 3600, 2),
                                 'windows_without_gap': fallbacks}
    rows = synthetic_meetings(settings, output, rng)
    clips += rows
    for dataset in sorted({r['dataset'] for r in rows}):
        part = [r for r in rows if r['dataset'] == dataset]
        report[dataset] = {'clips': len(part), 'hours': round(sum(r['duration'] for r in part) / 3600, 2)}
    speakers = {}
    for c in clips:
        speakers.setdefault(c['dataset'], {}).setdefault(c['reference_speakers'], 0)
        speakers[c['dataset']][c['reference_speakers']] += 1
    report['speaker_counts'] = speakers
    report['max_duration'] = max(c['duration'] for c in clips)
    manifest = output / 'clips.jsonl'
    manifest.write_text(''.join(json.dumps(c, ensure_ascii=False) + '\n' for c in clips))
    plan = json.loads(Path(settings['reference_plan']).read_text())
    plan['encoder_attention'] = dict(plan['encoder_attention'], n_window_infer=settings['n_window_infer'])
    plan.update(manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(), samples=len(clips),
                meetings=len({c['meeting_id'] for c in clips}),
                audio_hours=sum(c['duration'] for c in clips) / 3600, priority_ids=[],
                max_model_len=settings['max_model_len'],
                purpose='180 s deployment-length meeting benchmark (real zh/en + synthetic 6-10 speakers)')
    (output / 'plan.json').write_text(json.dumps(plan, indent=2, ensure_ascii=False) + '\n')
    (output / 'report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
