"""Evaluate several models on the same panels with the deployment stack.

Each model is served by ``open-audio-llm serve`` (vLLM HTTP) on one GPU of this
machine, every panel runs as an ``open-audio-llm eval`` task (AmphionEval, its
own W&B run) against that server, and the server is stopped before the next
model. Panels of a model that an earlier attempt already completed are reused
from ``reuse`` (copied summary, the original attempt stays the evidence).
"""
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import urllib.request

import yaml
from open_audio_llm.run_config import load_config


def recipe(template, folder):
    config = load_config(Path(template), folder / 'preview')
    config.pop('config_file', None)
    return config


def ready(url, name, process, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f'vLLM server exited {process.returncode}')
        try:
            with urllib.request.urlopen(f'{url}/v1/models', timeout=5) as response:
                if name in {m['id'] for m in json.load(response)['data']}:
                    return
        except OSError:
            pass
        time.sleep(10)
    raise TimeoutError(f'{url} did not serve {name}')


def latest_summary(folder):
    attempts = sorted((folder / 'attempts').glob('[0-9]*')) if (folder / 'attempts').exists() else []
    for attempt in reversed(attempts):
        path = attempt / 'artifacts/evaluation/summary.json'
        if path.exists() and 'coverage' not in json.loads(path.read_text()):
            return path
    return None


def evaluate_model(settings, name, spec, prefix):
    folder = Path(settings['output']) / 'models' / name
    folder.mkdir(parents=True, exist_ok=True)
    summaries, pending = {}, {}
    for panel, panel_spec in settings['panels'].items():
        previous = settings.get('reuse') and latest_summary(Path(settings['reuse']) / 'models' / name / panel)
        if previous:
            summaries[panel] = {'summary': json.loads(previous.read_text()), 'source': str(previous)}
        else:
            pending[panel] = panel_spec
    if not pending:
        return summaries
    with serve_model(settings, name, spec, folder) as (url, served):
        jobs = {}
        for panel, panel_spec in pending.items():
            config = recipe(panel_spec['template'], folder)
            config['task']['server_url'] = url
            summary = {'path': '{attempt}/artifacts/evaluation/summary.json'}
            config['task']['arguments'].update({'--model': str(spec['weights']), '--server-url': url,
                                                '--served-model': served,
                                                '--output_dir': {'path': '{attempt}/artifacts/evaluation'}})
            # Model-specific request settings, e.g. the plain output format of a single-speaker model.
            config['task']['arguments'].update(spec.get('arguments', {}))
            config['recording'] = {'attempts_dir': {'path': str(folder / panel / 'attempts')}, 'result': summary}
            config['tracking'].update(name=f'{prefix}-{name}-{panel}', metrics=[{'file': summary, 'prefix': 'evaluation'}])
            path = folder / f'{panel}.yaml'
            path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True))
            log = (folder / f'{panel}.log').open('a')
            jobs[panel] = (subprocess.Popen([sys.executable, '-m', 'open_audio_llm.cli', 'eval', '--config', str(path)],
                                            stdout=log, stderr=subprocess.STDOUT), log)
        failed = []
        for panel, (process, log) in jobs.items():
            code = process.wait()
            log.close()
            path = latest_summary(folder / panel)
            if code or path is None:
                failed.append(f'{panel} exited {code}')
                continue
            summaries[panel] = {'summary': json.loads(path.read_text()), 'source': str(path)}
            if settings.get('delete_request_audio'):
                # Per-request WAV copies of the panel audio; the panel itself stays in the dataset.
                subprocess.run(['rm', '-rf', str(path.parent / 'audio')], check=True)
        if failed:
            raise RuntimeError(f'{name}: ' + '; '.join(failed))
        return summaries


@contextmanager
def serve_model(settings, name, spec, folder):
    """Serve one model on the configured GPU and port; yields (url, served name), stops it after."""
    served, port = name, settings['port']
    url = f'http://127.0.0.1:{port}'
    serve = recipe(spec.get('serve_template', settings['serve_template']), folder)
    serve['task']['model'] = str(spec['weights'])
    serve['task']['arguments'].update({'--port': port, '--served-model-name': served})
    serve['runtime']['gpus'] = [settings['gpu']]
    serve['recording'] = {'attempts_dir': {'path': str(folder / 'serve/attempts')}}
    (folder / 'serve.yaml').write_text(yaml.safe_dump(serve, sort_keys=False, allow_unicode=True))
    # vLLM listens with SO_REUSEPORT: a second server on a busy port would split the requests.
    with socket.socket() as probe:
        if probe.connect_ex(('127.0.0.1', port)) == 0:
            raise RuntimeError(f'Port {port} already has a listener')
    with (folder / 'serve.log').open('a') as log:
        server = subprocess.Popen([sys.executable, '-m', 'open_audio_llm.cli', 'serve', '--config',
                                   str(folder / 'serve.yaml')], stdout=log, stderr=subprocess.STDOUT,
                                  start_new_session=True)
    try:
        ready(url, served, server, settings['server_startup_timeout'])
        yield url, served
    finally:
        if server.poll() is None:
            os.killpg(server.pid, signal.SIGTERM)
            server.wait(timeout=300)


def main():
    # A stopped run must still stop its vLLM server (serve_model's finally); SIGTERM would skip it.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    settings = yaml.safe_load(parser.parse_args().config.read_text())['parameters']
    output = Path(settings['output'])
    results_path = output / 'results.json'
    results = json.loads(results_path.read_text()) if results_path.exists() else {}
    for name, spec in settings['models'].items():
        if name in results:
            continue
        results[name] = {'path': str(spec['weights']), 'panels': evaluate_model(settings, name, spec, settings['tracking_prefix'])}
        results_path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + '\n')
        print(f'{name}: done', flush=True)


if __name__ == '__main__':
    main()
