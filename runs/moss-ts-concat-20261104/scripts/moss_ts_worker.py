"""AmphionEval JSONL worker: TS-ASR with MOSS-Transcribe-Diarize by prepending the enrollment.

Each request audio is what AmphionEval sends a TS-ASR model: the enrollment cut or
zero-padded to ``--enroll-seconds``, followed by the mixture. The worker splits it
there, inserts ``--gap-seconds`` of silence, sends the result to the official MOSS
vLLM server (/v1/audio/transcriptions, server default prompt, temperature 0) and
keeps the text of speaker ``--target-speaker`` from segments that start after the
enrollment. The full MOSS output is returned as ``raw_text`` for later analysis.

The server runs in this (the model's own) environment as a child process; its
version and command are written next to the predictions.
"""
import argparse
import io
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests
import soundfile as sf

# Same pattern as amphion_eval.parsers.parse_moss_transcribe_diarize: [start][Sxx]text[end].
SEGMENT = re.compile(
    r"\s*\[(?P<start>\d+(?:\.\d+)?)\]\[(?P<speaker>S\d+)\](?P<text>.*?)\[(?P<end>\d+(?:\.\d+)?)\]",
    re.DOTALL,
)

START_ATTEMPTS = 3


# A decode cut at the token limit (repetition loops) ends inside a segment: [start][Sxx]text...
PARTIAL = re.compile(r"\s*\[(?P<start>\d+(?:\.\d+)?)\]\[(?P<speaker>S\d+)\](?P<text>.*)$", re.DOTALL)


def parse(text):
    """Segments of a MOSS output; a truncated tail is kept so loops stay in the score, as in the meeting eval."""
    segments, position = [], 0
    while position < len(text):
        match = SEGMENT.match(text, position)
        if match is None:
            break
        segments.append({'speaker': match['speaker'], 'start': float(match['start']),
                         'end': float(match['end']), 'text': match['text'].strip()})
        position = match.end()
    rest = text[position:]
    truncated = bool(rest.strip())
    tail = PARTIAL.match(rest)
    if tail:
        segments.append({'speaker': tail['speaker'], 'start': float(tail['start']), 'end': None,
                         'text': tail['text'].strip()})
    return segments, truncated


def wait_ready(base, process, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f'vLLM server exited {process.returncode}')
        try:
            if requests.get(f'{base}/v1/models', timeout=5).ok:
                return
        except requests.RequestException:
            pass
        time.sleep(5)
    raise TimeoutError('vLLM server did not become ready')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--served-model-name', required=True)
    parser.add_argument('--gpu', required=True)
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--gpu-memory-utilization', required=True)
    parser.add_argument('--enroll-seconds', type=float, required=True)
    parser.add_argument('--gap-seconds', type=float, required=True)
    parser.add_argument('--target-speaker', required=True)
    parser.add_argument('--concurrency', type=int, required=True)
    parser.add_argument('--startup-timeout', type=float, required=True)
    parser.add_argument('--request-timeout', type=float, required=True)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    base = f'http://127.0.0.1:{args.port}'
    command = [sys.executable, '-m', 'vllm.entrypoints.cli.main', 'serve', args.model,
               '--served-model-name', args.served_model_name, '--port', str(args.port),
               '--trust-remote-code', '--gpu-memory-utilization', args.gpu_memory_utilization]
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': args.gpu}
    server = None
    try:
        # Other users' jobs on a shared card can free or take memory while vLLM profiles it at startup
        # (2026-11-05: OOM on GPU 2, profiling assertion on GPU 3); a start that dies is retried.
        for attempt in range(START_ATTEMPTS):
            with (args.output.parent / f'moss-server-{attempt + 1}.log').open('w') as log:
                server = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env,
                                          start_new_session=True)
            try:
                wait_ready(base, server, args.startup_timeout)
                break
            except RuntimeError:
                if attempt == START_ATTEMPTS - 1:
                    raise
                time.sleep(60)
        served = [m['id'] for m in requests.get(f'{base}/v1/models', timeout=10).json()['data']]
        if args.served_model_name not in served:
            raise RuntimeError(f'model not served: {served}')
        runtime = {'backend': 'vllm', 'server_version': requests.get(f'{base}/version', timeout=10).json()['version'],
                   'served_models': served, 'command': command, 'cuda_visible_devices': args.gpu,
                   'request': {'response_format': 'json', 'temperature': 0, 'prompt': 'server default'},
                   'enroll_seconds': args.enroll_seconds, 'gap_seconds': args.gap_seconds,
                   'target_speaker': args.target_speaker}
        (args.output.parent / 'moss-runtime.json').write_text(json.dumps(runtime, indent=2) + '\n')
        # Segments starting before the middle of the gap belong to the enrollment.
        boundary = args.enroll_seconds + args.gap_seconds / 2

        def transcribe(row):
            started = time.perf_counter()
            try:
                audio, rate = sf.read(row['audio'], dtype='float32')
                split = int(round(args.enroll_seconds * rate))
                joined = np.concatenate([audio[:split], np.zeros(int(round(args.gap_seconds * rate)), np.float32),
                                         audio[split:]])
                buffer = io.BytesIO()
                sf.write(buffer, joined, rate, format='WAV', subtype='PCM_16')
                response = requests.post(f'{base}/v1/audio/transcriptions',
                                         files={'file': ('audio.wav', buffer.getvalue(), 'audio/wav')},
                                         data={'model': args.served_model_name, 'response_format': 'json',
                                               'temperature': 0},
                                         timeout=args.request_timeout)
                response.raise_for_status()
                raw = response.json()['text']
                segments, truncated = parse(raw)
                text = ' '.join(s['text'] for s in segments
                                if s['speaker'] == args.target_speaker and s['start'] >= boundary)
                return {'id': row['id'], 'status': 'success', 'text': text, 'raw_text': raw,
                        'parse_truncated': truncated, 'elapsed_seconds': time.perf_counter() - started}
            except Exception as exc:  # noqa: BLE001 - a failed sample stays visible in the evaluation
                return {'id': row['id'], 'status': 'failed', 'error': f'{type(exc).__name__}: {exc}',
                        'elapsed_seconds': time.perf_counter() - started}

        done = 0
        with args.output.open('a') as stream, ThreadPoolExecutor(args.concurrency) as pool:
            for result in pool.map(transcribe, rows):
                stream.write(json.dumps(result, ensure_ascii=False) + '\n')
                stream.flush()
                done += 1
                if done % 500 == 0 or done == len(rows):
                    print(json.dumps({'completed': done, 'total': len(rows)}), flush=True)
    finally:
        if server is not None and server.poll() is None:
            os.killpg(server.pid, 15)
            server.wait(timeout=120)


if __name__ == '__main__':
    main()
