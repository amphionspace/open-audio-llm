"""Serve fixed models and run panels against each, optionally several times (decode-noise and sim-vs-real probes).

Reuses select_checkpoints.evaluate_model (same vLLM serving, AmphionEval meeting scoring and turn diagnostics as
the round-2 selectors), so every number here is directly comparable with the selection histories. Each job is one
served model; its panels run concurrently against that server, ``repeats`` copies of a panel being separate eval
executions (each its own W&B run)."""
import argparse
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import select_checkpoints as sc  # noqa: E402

LOCK = threading.Lock()


def compact(summary):
    """The fields the analysis reads: per-label cpER (with/without loops), loops, groups, overall, turn diagnostics."""
    out = {'labels': {}, 'all': {k: summary['all'][k] for k in ('cp_error_rate', 'repetition_loops',
                                                                 'speaker_count_accuracy', 'der') if k in summary['all']}}
    for label, values in summary.get('labels', {}).items():
        out['labels'][label] = {k: values[k] for k in ('cp_error_rate', 'cp_error_rate_without_loops',
                                                        'repetition_loops') if k in values}
    out['turn_diagnostics'] = summary.get('turn_diagnostics', {})
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    settings = yaml.safe_load(parser.parse_args().config.read_text())['parameters']
    state = Path(settings['output'])
    state.mkdir(parents=True, exist_ok=True)
    result_file = state / 'probe-results.json'
    results = json.loads(result_file.read_text()) if result_file.exists() else {}
    free = {slot: settings['port_base'] + slot for slot in range(settings['parallel'])}

    def run(job):
        panels, diagnostics = {}, {}
        for name, spec in job['panels'].items():
            for k in range(1, job.get('repeats', 1) + 1):
                panels[f'{name}-r{k}'] = {'template': spec['template']}
                diagnostics[f'{name}-r{k}'] = spec.get('turn_diagnostics', [])
        js = {**settings, 'panels': panels, 'turn_diagnostics': diagnostics,
              'serve_template': job.get('serve_template', settings['serve_template'])}
        with LOCK:
            slot = next(s for s, port in free.items() if port is not None)
            port, free[slot] = free[slot], None
        try:
            # gpu is picked from gpu_pool under the shared eval GPU lock when the server starts.
            summaries = sc.evaluate_model(js, job['model'], job['name'], None, port, settings['tracking_prefix'])
        finally:
            with LOCK:
                free[slot] = port
        with LOCK:
            results[job['name']] = {'model': str(job['model']), 'serve_template': str(js['serve_template']),
                                    'panels': {p: compact(s) for p, s in summaries.items()}}
            result_file.write_text(json.dumps(results, indent=2, ensure_ascii=False) + '\n')

    jobs = [j for j in settings['jobs'] if j['name'] not in results]
    with ThreadPoolExecutor(settings['parallel']) as pool:
        for future in [pool.submit(run, job) for job in jobs]:
            future.result()


if __name__ == '__main__':
    main()
