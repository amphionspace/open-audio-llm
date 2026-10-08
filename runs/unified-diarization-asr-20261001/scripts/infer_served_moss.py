"""Run MOSS-Transcribe-Diarize through its official vLLM server and keep raw outputs.

The server runs in the model's own environment as a separate process (the official
recommendation for CUDA hosts without SGLang support); this client only sends audio
to the OpenAI-compatible /v1/audio/transcriptions endpoint. Server version, model
registration and request parameters are recorded in runtime.json.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
import yaml


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
    parser.add_argument('--config', type=Path, required=True)
    settings = yaml.safe_load(parser.parse_args().config.read_text())['parameters']
    output = Path(settings['output'])
    output.mkdir(parents=True, exist_ok=True)
    clips = [json.loads(line) for line in Path(settings['clips']).read_text().splitlines()]
    audio_dir = Path(settings['clip_audio_dir'])
    base = f"http://127.0.0.1:{settings['port']}"
    command = [sys.executable, '-m', 'vllm.entrypoints.cli.main', 'serve', settings['model'],
               '--served-model-name', settings['served_model_name'], '--port', str(settings['port']),
               *settings['serve_arguments']]
    with (output / 'server.log').open('w') as log:
        server = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        wait_ready(base, server, settings['startup_timeout'])
        models = requests.get(f'{base}/v1/models', timeout=10).json()
        version = requests.get(f'{base}/version', timeout=10).json()
        served = [m['id'] for m in models.get('data', [])]
        if settings['served_model_name'] not in served:
            raise RuntimeError(f'model not served: {served}')
        import torch
        runtime = {'backend': 'vllm', 'server_version': version.get('version'), 'served_models': served,
                   'command': command, 'request': settings['request'], 'gpu': torch.cuda.get_device_name(0),
                   'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES')}
        (output / 'runtime.json').write_text(json.dumps(runtime, indent=2) + '\n')

        def transcribe(clip):
            path = audio_dir / Path(clip['audio']).name
            data = path.read_bytes()
            assert hashlib.sha256(data).hexdigest() == clip['audio_sha256'], path
            started = time.perf_counter()
            response = requests.post(f'{base}/v1/audio/transcriptions',
                                     files={'file': (path.name, data, 'audio/wav')},
                                     data={'model': settings['served_model_name'], **settings['request']},
                                     timeout=settings['request_timeout'])
            response.raise_for_status()
            body = response.json()
            return {'id': clip['id'], 'text': body['text'], 'usage': body.get('usage'),
                    'hit_token_limit': False, 'inference_seconds': time.perf_counter() - started}

        done = 0
        with (output / 'predictions.jsonl').open('w') as stream, \
                ThreadPoolExecutor(settings['concurrency']) as pool:
            for row in pool.map(transcribe, clips):
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')
                done += 1
                if done % 20 == 0 or done == len(clips):
                    print(json.dumps({'completed': done, 'total': len(clips)}), flush=True)
        (output / 'inference-complete.json').write_text(json.dumps({'rows': done, 'backend': 'vllm'}) + '\n')
    finally:
        os.killpg(server.pid, 15)
        server.wait(timeout=120)


if __name__ == '__main__':
    main()
