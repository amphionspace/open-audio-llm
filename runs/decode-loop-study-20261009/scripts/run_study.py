"""Serve each (model, decoding setting, repeat) with the deployment stack and run full eval sets.

One job = one ``open-audio-llm serve`` (vLLM HTTP) started with the job's decoding setting as
``--override-generation-config`` (absent for the deployed default), every listed eval set run by
AmphionEval as its own ``open-audio-llm eval`` execution with its own W&B run, then the server is
stopped. Jobs run in parallel, one per configured GPU; a GPU listed with ``after`` is used only once
that execution's status.json is completed or failed (it belongs to another running experiment).

The server logs every request's effective SamplingParams (``--enable-log-requests``). After the
evaluations each job checks that every request carried the configured repetition penalty and that
the request count covers all evaluated samples; a mismatch fails the job.

Finished jobs (``result.json``) are skipped when this attempt is resumed or when an earlier attempt
listed in ``reuse_jobs_dirs`` has them.
"""
import argparse
import itertools
import json
import os
from pathlib import Path
import queue
import re
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request

import yaml
from open_audio_llm.run_config import load_config

LOCK = threading.Lock()
SERVERS = set()


class ServerStartError(RuntimeError):
    """The GPU could not host a server; the worker stops taking jobs."""
REQUEST = re.compile(r'Received request (\S+): params: SamplingParams\((.*)\), lora_request')
FIELDS = ('repetition_penalty', 'frequency_penalty', 'presence_penalty', 'temperature', 'top_p', 'top_k',
          'max_tokens')


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


def free_mib(gpu):
    out = subprocess.run(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits', '-i',
                          str(gpu)], check=True, capture_output=True, text=True).stdout
    return int(out.strip())


def wait_for_gpu(settings, slot):
    after = slot.get('after')
    while after and json.loads(Path(after).read_text()).get('execution') not in {'completed', 'failed'}:
        time.sleep(settings['poll_seconds'])
    # vLLM reserves a fixed fraction of the whole card at startup; co-tenants must leave that free.
    while free_mib(slot['id']) < settings['min_free_mib']:
        time.sleep(settings['poll_seconds'])


def sampling_check(log_text, expected, samples):
    rows = []
    for match in REQUEST.finditer(log_text):
        params = dict(re.findall(r'(\w+)=([^,]+)', match.group(2)))
        rows.append({k: params.get(k) for k in FIELDS})
    observed = {k: sorted({r[k] for r in rows}) for k in FIELDS}
    want = str(float(expected.get('repetition_penalty', 1.0)))
    ok = len(rows) >= samples and observed['repetition_penalty'] == [want] and observed['temperature'] == ['0.0']
    overridden = [line for line in log_text.splitlines() if 'Default vLLM sampling parameters have been overridden' in line]
    return {'requests': len(rows), 'samples': samples, 'expected_repetition_penalty': float(want),
            'observed': observed, 'startup_override_log': overridden, 'passed': ok}


def run_job(settings, job, slot):
    folder = Path(settings['output']) / 'jobs' / job['key']
    folder.mkdir(parents=True, exist_ok=True)
    model, sampling = settings['models'][job['model']], settings['settings'][job['setting']]
    name, port = f"{job['model']}-{job['setting']}-r{job['repeat']}", slot['port']
    url = f'http://127.0.0.1:{port}'
    serve = recipe(settings['serve_template'], folder)
    serve['task']['model'] = model
    serve['task']['arguments'].update({'--port': port, '--served-model-name': name})
    if sampling:
        serve['task']['arguments']['--override-generation-config'] = sampling
    serve['runtime']['gpus'] = [slot['id']]
    serve['recording'] = {'attempts_dir': {'path': str(folder / 'serve/attempts')}}
    (folder / 'serve.yaml').write_text(yaml.safe_dump(serve, sort_keys=False, allow_unicode=True))
    # vLLM listens with SO_REUSEPORT: a second server on a busy port would split requests with it.
    with socket.socket() as probe:
        if probe.connect_ex(('127.0.0.1', port)) == 0:
            raise RuntimeError(f'Port {port} already has a listener')
    with (folder / 'serve.log').open('a') as log:
        server = subprocess.Popen([sys.executable, '-m', 'open_audio_llm.cli', 'serve', '--config',
                                   str(folder / 'serve.yaml')], stdout=log, stderr=subprocess.STDOUT,
                                  start_new_session=True)
    SERVERS.add(server)
    try:
        try:
            ready(url, name, server, settings['server_startup_timeout'])
        except (RuntimeError, TimeoutError) as error:
            raise ServerStartError(str(error)) from error
        jobs = {}
        for panel in job['panels']:
            config = recipe(settings['panels'][panel]['template'], folder)
            config['task']['server_url'] = url
            summary = {'path': '{attempt}/artifacts/evaluation/summary.json'}
            config['task']['arguments'].update({'--model': model, '--server-url': url, '--served-model': name,
                                                '--output_dir': {'path': '{attempt}/artifacts/evaluation'}})
            config['recording'] = {'attempts_dir': {'path': str(folder / panel / 'attempts')}, 'result': summary}
            config['tracking'].update(name=f"{settings['tracking_prefix']}-{name}-{panel}",
                                      metrics=[{'file': summary, 'prefix': 'evaluation'}])
            # Recorded in the eval snapshot and its W&B config; the setting itself acts on the server.
            config['study'] = {'model': job['model'], 'setting': job['setting'], 'repeat': job['repeat'],
                               'override_generation_config': sampling or None, 'serve_config': str(folder / 'serve.yaml')}
            path = folder / f'{panel}.yaml'
            path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True))
            log = (folder / f'{panel}.log').open('a')
            jobs[panel] = (subprocess.Popen([sys.executable, '-m', 'open_audio_llm.cli', 'eval', '--config',
                                             str(path)], stdout=log, stderr=subprocess.STDOUT), log)
        panels, failed = {}, []
        for panel, (process, log) in jobs.items():
            code = process.wait()
            log.close()
            attempt = max((folder / panel / 'attempts').glob('[0-9]*'))
            evaluation = attempt / 'artifacts/evaluation'
            entry = {'attempt': str(attempt), 'exit_code': code, 'rescored': False,
                     'scores_file': str(evaluation / 'meeting-scores.jsonl'), 'summary_file': str(evaluation / 'summary.json')}
            if code:
                # AmphionEval 0.7.0 drops every answer after one containing U+2028 (splitlines reader);
                # rescore.py scores the complete server answers with AmphionEval's own functions.
                rescore = subprocess.run([settings['rescore_python'], '-I', settings['rescore_script'], str(evaluation)],
                                         capture_output=True, text=True)
                (folder / f'{panel}-rescore.log').write_text(rescore.stdout + rescore.stderr)
                if rescore.returncode:
                    failed.append(f'{panel} exited {code}')
                    continue
                entry.update(rescored=True, scores_file=str(evaluation / 'meeting-scores-rescored.jsonl'),
                             summary_file=str(evaluation / 'summary-rescored.json'))
            summary = json.loads(Path(entry['summary_file']).read_text())
            entry.update(summary=summary, coverage={'total': sum(v['samples'] for v in summary['labels'].values())})
            panels[panel] = entry
            if settings.get('delete_request_audio'):
                # Per-request WAV copies of the eval audio; the datasets themselves are untouched.
                subprocess.run(['rm', '-rf', str(evaluation / 'audio')], check=True)
        if failed:
            raise RuntimeError(f"{job['key']}: {'; '.join(failed)}")
    finally:
        if server.poll() is None:
            os.killpg(server.pid, signal.SIGTERM)
            server.wait(timeout=300)
        SERVERS.discard(server)
    # The serve execution writes vLLM's own output to its process log.
    text = (max((folder / 'serve/attempts').glob('[0-9]*')) / 'logs/process.log').read_text()
    check = sampling_check(text, sampling, sum(p['coverage']['total'] for p in panels.values()))
    result = {**job, 'model_path': model, 'override_generation_config': sampling or None, 'gpu': slot['id'],
              'port': port, 'sampling_verification': check, 'panels': panels}
    (folder / 'sampling-verification.json').write_text(json.dumps(check, indent=2) + '\n')
    if not check['passed']:
        raise RuntimeError(f"{job['key']}: effective sampling parameters do not match: {check['observed']}")
    (folder / 'result.json').write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    return result


def expand(settings):
    jobs = {}
    for block in settings['jobs']:
        for model, setting, repeat in itertools.product(block['models'], block['settings'], block['repeats']):
            key = f'{model}/{setting}/r{repeat}'
            job = jobs.setdefault(key, {'key': key, 'model': model, 'setting': setting, 'repeat': repeat, 'panels': []})
            job['panels'] += [p for p in block['panels'] if p not in job['panels']]
    return list(jobs.values())


def progress(settings, jobs, results, failures):
    row = {'jobs_total': len(jobs), 'jobs_done': len(results), 'jobs_failed': len(failures)}
    for key, result in results.items():
        base = key.replace('/', '-')
        row[f'{base}/sampling/repetition_penalty'] = result['sampling_verification']['expected_repetition_penalty']
        row[f'{base}/sampling/requests'] = result['sampling_verification']['requests']
        for panel, values in result['panels'].items():
            for label, metrics in values['summary']['labels'].items():
                for field in ('repetition_loops', 'cp_error_rate', 'cp_error_rate_without_loops', 'der',
                              'speaker_count_accuracy'):
                    row[f'{base}/{panel}/{label}/{field}'] = metrics[field]
    out = Path(settings['output'])
    (out / 'study-progress.json').write_text(json.dumps(row, indent=2) + '\n')
    (out / 'failures.json').write_text(json.dumps(failures, indent=2, ensure_ascii=False) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    settings = yaml.safe_load(parser.parse_args().config.read_text())['parameters']
    out = Path(settings['output'])
    out.mkdir(parents=True, exist_ok=True)
    jobs = expand(settings)
    (out / 'jobs.json').write_text(json.dumps(jobs, indent=2) + '\n')
    results, failures = {}, {}
    pending = queue.Queue()
    for job in jobs:
        # Finished jobs of this attempt (resume) or of earlier attempts named in reuse_jobs_dirs.
        done = [d / job['key'] / 'result.json' for d in [out / 'jobs', *map(Path, settings.get('reuse_jobs_dirs', []))]]
        done = [d for d in done if d.is_file()]
        if done:
            results[job['key']] = {**json.loads(done[0].read_text()), 'result_file': str(done[0])}
        else:
            pending.put(job)
    progress(settings, jobs, results, failures)

    def worker(slot):
        while not pending.empty():
            wait_for_gpu(settings, slot)
            try:
                job = pending.get_nowait()
            except queue.Empty:
                return
            try:
                result = run_job(settings, job, slot)
            except Exception as error:  # noqa: BLE001 - recorded; the job is not retried here
                with LOCK:
                    failures[job['key']] = {'gpu': slot['id'], 'error': f'{type(error).__name__}: {error}'}
                    progress(settings, jobs, results, failures)
                if isinstance(error, ServerStartError):
                    return
                continue
            with LOCK:
                results[job['key']] = result
                progress(settings, jobs, results, failures)

    def stop(signum, frame):
        # Servers run in their own sessions so evaluations cannot signal them; stop them explicitly.
        for server in list(SERVERS):
            if server.poll() is None:
                os.killpg(server.pid, signal.SIGTERM)
        os._exit(128 + signum)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    threads = [threading.Thread(target=worker, args=(slot,)) for slot in settings['gpus']]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if failures or len(results) != len(jobs):
        raise SystemExit(f'{len(results)}/{len(jobs)} jobs finished; failures: {sorted(failures)}')


if __name__ == '__main__':
    main()
