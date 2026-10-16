"""Whole-session speaker-attributed transcription of real meetings (paper protocol).

The system under test is the served model plus this reference-free long-form
pipeline; nothing below reads the references except the scorer:

1. Each session's first far-field channel (AliMeeting, AISHELL-4 channel 0; AMI
   SDM) is cut into windows of about ``window.target`` seconds at the quietest
   100 ms frame within ``window.search`` seconds of the target.
2. Every window is sent to the vLLM server with the evaluation timestamp prompt
   (temperature 0); the output lines ``[S1][0.00-2.50] text`` are shifted by the
   window offset. Text outside a well-formed line is kept (attached to the
   previous line's speaker) so no recognized word is dropped.
3. Speakers are linked across windows: each (window, speaker) with at least
   ``embedding.min_seconds`` of speech that no other speaker of the window
   overlaps gets a WeSpeaker ECAPA embedding of that speech (a JSONL worker in
   its own environment); average-linkage clustering on cosine
   similarity merges them, never merging two speakers of the same window, while
   the similarity exceeds the threshold. Shorter speakers stay unlinked.
4. The threshold is chosen per model on ``tune_set`` (a development set) by
   whole-session cpER, then applied unchanged to every set.

A plain-transcript model (``plain: true``) gets the Qwen3-ASR transcription
request on the same windows and is scored speaker-agnostically only.
"""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import gzip
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import urllib.request

import numpy as np
import soundfile as sf
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from run_benchmark import serve_model  # noqa: E402

RATE = 16000
# Frozen evaluation prompt (amphion_eval.integrations.open_audio_llm.timestamps.TIMESTAMP_SYSTEM).
TIMESTAMP_SYSTEM = (
    "Transcribe every utterance from every speaker, including overlapping speech. "
    "Output one line per utterance as [S1][0.00-2.50] text, with start and end "
    "times in seconds from the beginning of the audio. Sort lines by start time. "
    "Keep the same speaker ID across turns; assign IDs by first speech. "
    "Use the beginning of the first spoken word and the end of the last spoken "
    "word as the utterance boundaries, excluding leading and trailing silence."
)
LINE = re.compile(r'^\[S(\d+)\]\[(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?)\]\s*(.*)$')
TAG = re.compile(r'\[S\d+\]|\[\d+(?:\.\d+)?(?:-\d+(?:\.\d+)?)?\]')


def jsonl(path):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path, items):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as stream:
        for item in items:
            stream.write(json.dumps(item, ensure_ascii=False) + '\n')


def artifacts(registry, spec):
    """Recording and supervision manifests of a registered split (aidc_data root only)."""
    roots = json.loads(Path(registry['roots']).read_text())
    for path in sorted(Path(registry['catalog']).glob('*.yaml')):
        for entry in yaml.safe_load(path.read_text()):
            if (entry['dataset_id'], entry['version']) != (spec['dataset_id'], spec['version']):
                continue
            named = {a['name']: Path(roots[a['root_alias']]) / a['relative_path'] for a in entry['artifacts']}
            split = entry['splits'][spec['split']]['artifacts']
            return [named[n] for n in split['recordings']], [named[n] for n in split['supervisions']]
    raise KeyError(spec)


def cut_points(audio, target, search):
    """Window boundaries at the quietest 100 ms frame near each target length."""
    frame = RATE // 10
    energy = np.sqrt(np.mean(audio[:len(audio) // frame * frame].reshape(-1, frame) ** 2, axis=1))
    points, start = [0], 0
    while len(audio) - start > (target + search) * RATE:
        low, high = (start // frame) + int((target - search) * 10), (start // frame) + int((target + search) * 10)
        start = (low + int(np.argmin(energy[low:high]))) * frame + frame // 2
        points.append(start)
    return points + [len(audio)]


def prepare(settings, shared):
    """Session audio (first channel), windows and references; shared by every model."""
    done = shared / 'prepared.json'
    if done.exists():
        return json.loads(done.read_text())
    window = settings['window']
    windows, references = [], []
    for name, spec in settings['sets'].items():
        recordings, supervisions = artifacts(settings['registry'], spec)
        turns = {}
        for path in supervisions:
            for s in jsonl(path):
                turns.setdefault(s['recording_id'], []).append(
                    {'speaker': s['speaker'], 'start': s['start'], 'end': s['start'] + s['duration'], 'text': s['text']})
        recordings = [r for path in recordings for r in jsonl(path)]
        if spec.get('sessions'):
            # A development subset: the first N sessions by id (deterministic, no reference is read to choose).
            recordings = sorted(recordings, key=lambda r: r['id'])[:spec['sessions']]
        for recording in recordings:
            session = recording['id']
            source = recording['sources'][0]['source']
            audio, rate = sf.read(source, dtype='float32', always_2d=True)
            assert rate == RATE, source
            audio = audio[:, spec.get('channel', 0)]
            folder = shared / 'audio' / name
            folder.mkdir(parents=True, exist_ok=True)
            sf.write(folder / f'{session}.wav', audio, RATE)
            references.append({'set': name, 'session': session, 'language': spec['language'],
                               'duration': len(audio) / RATE, 'turns': turns[session]})
            points = cut_points(audio, window['target'], window['search'])
            for index, (a, b) in enumerate(zip(points, points[1:])):
                path = shared / 'windows' / name / f'{session}_{index:03d}.wav'
                path.parent.mkdir(parents=True, exist_ok=True)
                sf.write(path, audio[a:b], RATE)
                windows.append({'set': name, 'session': session, 'index': index, 'offset': a / RATE,
                                'duration': (b - a) / RATE, 'path': str(path),
                                'session_audio': str(folder / f'{session}.wav')})
            print(f'{name}/{session}: {len(points) - 1} windows', flush=True)
    write_jsonl(shared / 'windows.jsonl', windows)
    write_jsonl(shared / 'references.jsonl', references)
    summary = {'windows': str(shared / 'windows.jsonl'), 'references': str(shared / 'references.jsonl')}
    done.write_text(json.dumps(summary) + '\n')
    return summary


def request(url, served, window, plain, max_tokens):
    audio = base64.b64encode(Path(window['path']).read_bytes()).decode('ascii')
    payload = {'model': served, 'temperature': 0.0, 'max_tokens': max_tokens, 'messages': [
        {'role': 'system', 'content': '' if plain else TIMESTAMP_SYSTEM},
        {'role': 'user', 'content': [{'type': 'input_audio', 'input_audio': {'data': audio, 'format': 'wav'}}]}]}
    data = json.dumps(payload).encode()
    req = urllib.request.Request(f'{url}/v1/chat/completions', data=data, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=3600) as response:
        body = json.load(response)
    return {**window, 'raw': body['choices'][0]['message']['content'],
            'finish_reason': body['choices'][0].get('finish_reason')}


def segments(output, plain):
    """Absolute-time segments of one window's output, keyed by window-local speaker."""
    text = output['raw'].split('<asr_text>', 1)[-1]
    offset, end = output['offset'], output['offset'] + output['duration']
    if plain:
        return [{'speaker': 'all', 'start': offset, 'end': end, 'text': text.strip()}]
    result = []
    for line in text.splitlines():
        if not line.strip():
            continue
        match = LINE.match(line.strip())
        if match:
            start = min(offset + float(match[2]), end)
            result.append({'speaker': f'S{int(match[1])}', 'start': start,
                           'end': max(start, min(offset + float(match[3]), end)), 'text': match[4].strip()})
        elif result:
            result[-1]['text'] += ' ' + TAG.sub(' ', line).strip()
        else:
            result.append({'speaker': 'S0', 'start': offset, 'end': offset, 'text': TAG.sub(' ', line).strip()})
    return result


def link(nodes, threshold):
    """Average-linkage clustering with cannot-link inside a window; returns node -> cluster."""
    clusters = [[i] for i in range(len(nodes))]
    embedded = [n['embedding'] is not None for n in nodes]
    vectors = np.array([n['embedding'] if n['embedding'] is not None else np.zeros(len(next(
        (m['embedding'] for m in nodes if m['embedding'] is not None), [0]))) for n in nodes])
    if embedded.count(True):
        vectors = vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-9)
    similarity = vectors @ vectors.T
    while True:
        best, pair = threshold, None
        for a in range(len(clusters)):
            for b in range(a + 1, len(clusters)):
                left, right = clusters[a], clusters[b]
                if not all(embedded[i] for i in left + right):
                    continue
                if {nodes[i]['window'] for i in left} & {nodes[i]['window'] for i in right}:
                    continue
                score = float(np.mean(similarity[np.ix_(left, right)]))
                if score > best:
                    best, pair = score, (a, b)
        if pair is None:
            break
        clusters[pair[0]] += clusters.pop(pair[1])
    return {i: c for c, members in enumerate(clusters) for i in members}


def link_two_stage(nodes, speech, high, small_fraction, join):
    """Pure clusters at a high threshold; then each cluster with less than ``small_fraction`` of the session's
    speech joins the most similar large cluster it shares no window with, if the average similarity exceeds ``join``."""
    assignment = link(nodes, high)
    clusters = {}
    for i, c in assignment.items():
        clusters.setdefault(c, []).append(i)
    vectors = {i: np.asarray(n['embedding']) / max(np.linalg.norm(n['embedding']), 1e-9)
               for i, n in enumerate(nodes) if n['embedding'] is not None}
    total = sum(speech) or 1.0
    size = {c: sum(speech[i] for i in members) for c, members in clusters.items()}
    large = [c for c in clusters if size[c] >= small_fraction * total]
    for c in sorted(clusters, key=lambda c: size[c]):
        members = [i for i in clusters[c] if i in vectors]
        if c in large or not members:
            continue
        windows = {nodes[i]['window'] for i in clusters[c]}
        best, target = join, None
        for b in large:
            if windows & {nodes[i]['window'] for i in clusters[b]}:
                continue
            score = float(np.mean([vectors[i] @ vectors[j] for i in members for j in clusters[b] if j in vectors]))
            if score > best:
                best, target = score, b
        if target is not None:
            clusters[target] += clusters.pop(c)
    return {i: c for c, members in clusters.items() for i in members}


def hypotheses(outputs, embeddings, plain, threshold, min_seconds, two_stage=None):
    sessions = {}
    for output in outputs:
        sessions.setdefault((output['set'], output['session']), []).append(output)
    result = []
    for (name, session), items in sorted(sessions.items()):
        if plain:
            segs = [s for o in sorted(items, key=lambda o: o['index']) for s in segments(o, True)]
            result.append({'set': name, 'session': session, 'plain': True, 'segments': segs})
            continue
        nodes, segs, speech = [], [], []
        for output in sorted(items, key=lambda o: o['index']):
            local = {}
            for s in segments(output, False):
                key = (output['index'], s['speaker'])
                if key not in local:
                    local[key] = len(nodes)
                    node_id = f"{name}/{session}/{output['index']}/{s['speaker']}"
                    nodes.append({'window': output['index'], 'id': node_id, 'embedding': embeddings.get(node_id)})
                    speech.append(0.0)
                speech[local[key]] += s['end'] - s['start']
                segs.append({**s, 'node': local[key]})
        assignment = link(nodes, threshold) if two_stage is None else link_two_stage(nodes, speech, *two_stage)
        result.append({'set': name, 'session': session, 'segments': [
            {**s, 'speaker': f'spk{assignment[s.pop("node")]}'} for s in segs]})
    return result


def solo(segs, speaker, shortest):
    """Spans of one speaker's segments that no other speaker's segment overlaps."""
    spans = []
    for s in segs:
        if s['speaker'] != speaker:
            continue
        parts = [[s['start'], s['end']]]
        for other in segs:
            if other['speaker'] == speaker:
                continue
            parts = [piece for a, b in parts for piece in (
                [[a, b]] if other['end'] <= a or other['start'] >= b else
                [p for p in ([a, other['start']], [other['end'], b]) if p[1] > p[0]])]
        spans += [p for p in parts if p[1] - p[0] >= shortest]
    return spans


def embed(settings, folder, outputs):
    """Embeddings of every (window, speaker) with enough non-overlapped speech, by the worker environment."""
    path = folder / 'embeddings.jsonl'
    if path.exists():
        return {r['id']: r['embedding'] for r in jsonl(path)}
    rows = []
    for output in outputs:
        segs = segments(output, False)
        # Overlapped speech mixes voices into the embedding (dev cpER 33.0 -> 22.8 without it).
        speech = {k: solo(segs, k, settings['embedding']['shortest_piece']) for k in {s['speaker'] for s in segs}}
        for speaker, spans in speech.items():
            if sum(b - a for a, b in spans) >= settings['embedding']['min_seconds']:
                rows.append({'id': f"{output['set']}/{output['session']}/{output['index']}/{speaker}",
                             'audio': output['session_audio'], 'segments': spans})
    rows.sort(key=lambda r: r['audio'])
    write_jsonl(folder / 'embedding-input.jsonl', rows)
    spec = settings['embedding']
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(spec['gpu'])} if 'gpu' in spec else None
    subprocess.run([spec['python'], spec['script'], spec['model'], str(folder / 'embedding-input.jsonl'),
                    str(path), '--max-seconds', str(spec['max_seconds'])], check=True, env=env)
    return {r['id']: r['embedding'] for r in jsonl(path)}


def score(settings, references, hypotheses_path, output):
    spec = settings['scorer']
    env = {'PYTHONPATH': spec['pythonpath'], 'PATH': '/usr/bin:/bin'}
    subprocess.run([spec['python'], spec['script'], references, str(hypotheses_path), str(output)], check=True, env=env)
    return json.loads(Path(output).read_text())


def evaluate(settings, name, spec, prepared):
    folder = Path(settings['output']) / 'models' / name
    folder.mkdir(parents=True, exist_ok=True)
    plain = spec.get('plain', False)
    windows = jsonl(prepared['windows'])
    outputs_path = folder / 'outputs.jsonl'
    previous = settings.get('reuse') and Path(settings['reuse']) / 'models' / name
    if previous and not outputs_path.exists():
        # Model outputs and embeddings do not depend on linking or scoring; the earlier attempt stays as evidence.
        for file in settings.get('reuse_files', ['outputs.jsonl']):
            if (previous / file).exists():
                shutil.copy2(previous / file, folder / file)
    outputs = jsonl(outputs_path) if outputs_path.exists() else []
    missing = [w for w in windows if (w['set'], w['session'], w['index']) not in
               {(o['set'], o['session'], o['index']) for o in outputs}]
    if missing:
        with serve_model(settings, name, spec, folder) as (url, served):
            with ThreadPoolExecutor(settings['request']['concurrency']) as pool, outputs_path.open('a') as sink:
                for output in pool.map(lambda w: request(url, served, w, plain, settings['request']['max_tokens']),
                                       missing):
                    sink.write(json.dumps(output, ensure_ascii=False) + '\n')
                    sink.flush()
        outputs = jsonl(outputs_path)
    tune = settings.get('tune_sets') or [settings['tune_set']]
    if plain:
        write_jsonl(folder / 'hypotheses.jsonl', hypotheses(outputs, {}, True, 0, 0))
        return {'threshold': None, 'scores': score(settings, prepared['references'], folder / 'hypotheses.jsonl',
                                                    folder / 'scores.json')}
    embeddings = embed(settings, folder, outputs)
    two = settings.get('two_stage')
    grid = ({f'{h:.2f}/{f:.2f}': (h, f, two['join']) for h in two['high'] for f in two['small_fraction']} if two
            else {f'{threshold:.2f}': threshold for threshold in settings['thresholds']})
    candidates = []
    for variant, value in grid.items():
        args = (0.0, 0, value) if two else (value, 0)
        for h in hypotheses([o for o in outputs if o['set'] in tune], embeddings, False, *args):
            candidates.append({**h, 'variant': variant})
    write_jsonl(folder / 'tuning-hypotheses.jsonl', candidates)
    references = [r for r in jsonl(prepared['references']) if r['set'] in tune]
    write_jsonl(folder / 'tuning-references.jsonl', references)
    sets = score(settings, str(folder / 'tuning-references.jsonl'), folder / 'tuning-hypotheses.jsonl',
                 folder / 'tuning-scores.json')['sets']
    # Micro over the development sets: errors over reference units of all their sessions.
    tuning = {v: sum(sets[s][v]['cp_error_rate'] * sets[s][v]['units'] for s in tune) /
              sum(sets[s][v]['units'] for s in tune) for v in grid}
    chosen = min(tuning, key=tuning.get)
    args = (0.0, 0, grid[chosen]) if two else (grid[chosen], 0)
    write_jsonl(folder / 'hypotheses.jsonl', hypotheses(outputs, embeddings, False, *args))
    scores = score(settings, prepared['references'], folder / 'hypotheses.jsonl', folder / 'scores.json')
    return {'threshold': chosen, 'tuning': tuning, 'scores': scores}


def main():
    # A stopped run must still stop its vLLM server (serve_model's finally); SIGTERM would skip it.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    settings = yaml.safe_load(parser.parse_args().config.read_text())['parameters']
    output = Path(settings['output'])
    prepared = prepare(settings, Path(settings['shared']))
    results_path = output / 'results.json'
    results = json.loads(results_path.read_text()) if results_path.exists() else {}
    for name, spec in settings['models'].items():
        if name in results:
            continue
        result = evaluate(settings, name, spec, prepared)
        sets = {k: {f: v[''][f] for f in ('cp_error_rate', 'agnostic_error_rate', 'der', 'speaker_count_mean_abs_error',
                                          'sessions')} for k, v in result['scores']['sets'].items()}
        results[name] = {'weights': str(spec['weights']), 'threshold': result['threshold'],
                         'tuning': result.get('tuning'), 'sets': sets}
        results_path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + '\n')
        # Flat metrics for W&B.
        flat = {f'{name}/{k}/{f}': v for k, m in sets.items() for f, v in m.items() if v is not None}
        with (output / 'metrics.jsonl').open('a') as sink:
            sink.write(json.dumps(flat) + '\n')
        print(f'{name}: {json.dumps(sets)}', flush=True)


if __name__ == '__main__':
    main()
