"""Select checkpoints with the deployment stack while training runs elsewhere.

Each saved checkpoint (and the start model, as the step-0 baseline) is served
with ``open-audio-llm serve`` (vLLM HTTP, full-parameter weights as saved) on a
free GPU of this machine, every selection panel is evaluated by AmphionEval
(``ae open-audio-llm meeting`` as an ``open-audio-llm eval`` task with its own
W&B run), and the server is stopped. Checkpoints are evaluated in parallel, one
per configured GPU, so selection keeps up with training.

The decision is recomputed in step order after every evaluation:

- stop when the last ``patience`` checkpoints all regress a retention metric
  by more than its tolerance, or all exceed the baseline repetition
  loops by more than ``loops.max_increase`` (written to ``stop_request`` in a
  directory this machine can write; the training callback stops every rank);
- the best checkpoint minimizes the target metric among checkpoints that keep
  every retention metric within tolerance and loops within the allowed increase.

``retention`` maps names to metric specs, each with its own tolerance. A spec
with ``source`` reads a single-speaker ASR panel (``ae open-audio-llm ts-asr``
summary): the pooled error rate of the entries whose key starts with ``source``.

After the training attempt completes, the chosen checkpoint runs the final
reporting panels. Execution completion and quality are recorded separately.
"""
import argparse
import difflib
import shutil
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request

import numpy as np
import yaml
from open_audio_llm.run_config import load_config
from rapidfuzz.distance import Indel, Levenshtein
from scipy.optimize import linear_sum_assignment

LOCK = threading.Lock()


def complete_checkpoints(training, interval, max_steps):
    """Saved checkpoints, local or already uploaded and removed by the upload callback."""
    found = {}
    uploads = training / 'storage-uploads.json'
    if uploads.is_file():
        for row in json.loads(uploads.read_text()):
            found[int(Path(row['checkpoint']).name.split('-')[1])] = Path(row['checkpoint'])
    for path in training.glob('checkpoint-[0-9]*'):
        # The sampler state is written after the weights and optimizer, so it marks a finished save.
        if (path / 'catalog_sampler.json').is_file():
            found[int(path.name.split('-')[1])] = path
    for step, path in sorted(found.items()):
        if step % interval == 0 or step == max_steps:
            yield step, path


# Files vLLM needs; optimizer, scheduler and RNG states stay with the training attempt.
SKIPPED = ('optimizer.pt', 'scheduler.pt', 'training_args.bin', 'rng_state')


def complete_weights(folder):
    index = folder / 'model.safetensors.index.json'
    if not (folder / 'config.json').is_file() or not index.is_file():
        return False
    return all((folder / name).is_file() for name in set(json.loads(index.read_text())['weight_map'].values()))


def materialize(settings, checkpoint, step):
    """Copy the weights into this attempt so the upload callback cannot delete them mid-evaluation.

    ACP writes as root: the training directory is read-only here, and checkpoints removed after
    upload are pulled back from object storage (weights only were uploaded).
    """
    target = Path(settings['output']) / 'models' / f'checkpoint-{step}'
    if (target / 'complete.json').is_file():
        return target
    shutil.rmtree(target, ignore_errors=True)
    target.mkdir(parents=True)
    deadline = time.monotonic() + settings['readable_timeout']
    while checkpoint.is_dir() and not all(os.access(p, os.R_OK) for p in checkpoint.glob('model*.safetensors')):
        # The training callback adds read bits right after the save.
        if time.monotonic() > deadline:
            raise PermissionError(f'{checkpoint} weights stayed unreadable')
        time.sleep(10)
    try:
        for path in checkpoint.iterdir():
            if path.is_file() and not path.name.startswith(SKIPPED):
                shutil.copy2(path, target / path.name)
        source = 'local'
        if not complete_weights(target):
            raise FileNotFoundError(checkpoint)
    except FileNotFoundError:
        # Removed (possibly while copying) by the upload callback after a confirmed upload.
        shutil.rmtree(target)
        target.mkdir(parents=True)
        from open_audio_llm.storage import pull, remote_path
        pull(remote_path(checkpoint, settings['storage']), str(target), settings['storage'])
        source = 'object_storage'
    if not complete_weights(target):
        raise RuntimeError(f'Incomplete weights for {checkpoint} from {source}')
    (target / 'complete.json').write_text(json.dumps({'checkpoint': str(checkpoint), 'source': source}) + '\n')
    return target


def recipe(template, folder, preview):
    config = load_config(Path(template), preview)
    config.pop('config_file', None)
    folder.mkdir(parents=True, exist_ok=True)
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


TURN = re.compile(r'^\[(S\d+)\]\[([\d.]+)-([\d.]+)\]\s*(.*)$')
UNIT = re.compile(r'[\u4e00-\u9fffA-Za-z0-9]')
# Reference turns are matched to hypothesis text starting or ending within this many seconds (text_errors).
TEXT_MARGIN = 2.0


def turns_of(text):
    out = []
    for line in (text or '').splitlines():
        m = TURN.match(line.strip())
        if m:
            out.append((m[1], float(m[2]), float(m[3]), ''.join(UNIT.findall(m[4])).lower()))
    return sorted(out, key=lambda x: x[1])


def align_owners(ref, hyp, owner):
    """Reference turn index charged with each edit of ref -> hyp (insertions go to the preceding reference unit)."""
    n, m = len(ref), len(hyp)
    d = [list(range(m + 1))] + [[i] + [0] * m for i in range(1, n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]), d[i - 1][j] + 1, d[i][j - 1] + 1)
    i, j, charged = n, m, []
    while i or j:
        if i and j and d[i][j] == d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            if ref[i - 1] != hyp[j - 1]:
                charged.append(owner[i - 1])
            i, j = i - 1, j - 1
        elif i and d[i][j] == d[i - 1][j] + 1:
            charged.append(owner[i - 1]); i -= 1
        else:
            charged.append(owner[max(i - 1, 0)] if owner else None); j -= 1
    return charged


def duplicate_pairs(turns):
    """Lines of two speakers with overlapping times and near-identical text (not counted as repetition loops)."""
    count = 0
    for k, a in enumerate(turns):
        for b in turns[k + 1:k + 4]:
            if a[0] == b[0] or min(len(a[3]), len(b[3])) < 4:
                continue
            if min(a[2], b[2]) - max(a[1], b[1]) >= 0.5 * min(a[2] - a[1], b[2] - b[1]) and \
                    difflib.SequenceMatcher(None, a[3], b[3]).ratio() >= 0.8:
                count += 1
    return count


def speaker_mapping(ref, hyp):
    """Reference -> hypothesis speaker mapping with minimum total edit distance (as cpER)."""
    rs, hs = sorted({t[0] for t in ref}), sorted({t[0] for t in hyp})
    rtext = {s: ''.join(t[3] for t in ref if t[0] == s) for s in rs}
    htext = {s: ''.join(t[3] for t in hyp if t[0] == s) for s in hs}
    mapping = {}
    if hs:
        # Gain of mapping r -> h over leaving both unmapped (r all deleted, h all inserted).
        cost = np.array([[Levenshtein.distance(rtext[r], htext[h]) - len(rtext[r]) - len(htext[h]) for h in hs]
                         for r in rs])
        for i, j in zip(*linear_sum_assignment(cost)):
            if cost[i, j] < 0:
                mapping[rs[i]] = hs[j]
    return mapping, rtext, htext


def speaker_errors(ref, hyp):
    """Edits charged to each reference turn when each reference speaker is aligned with its mapped hypothesis
    speaker (the cpER alignment; immune to how interrupted turns are ordered across speakers)."""
    mapping, rtext, htext = speaker_mapping(ref, hyp)
    errors = [0] * len(ref)
    for r in rtext:
        owner = [k for k, t in enumerate(ref) if t[0] == r for _ in t[3]]
        for k in align_owners(rtext[r], htext.get(mapping.get(r), ''), owner):
            if k is not None:
                errors[k] += 1
    return errors


def semi_global(ref, hyp):
    """Edit distance of ref against the best-matching substring of hyp."""
    if not hyp:
        return len(ref)
    prev = [0] * (len(hyp) + 1)
    for i in range(1, len(ref) + 1):
        cur = [i] + [0] * len(hyp)
        for j in range(1, len(hyp) + 1):
            cur[j] = min(prev[j - 1] + (ref[i - 1] != hyp[j - 1]), prev[j] + 1, cur[j - 1] + 1)
        prev = cur
    return min(prev)


def text_errors(ref, hyp, margin=TEXT_MARGIN):
    """Lower bound of each reference turn's recognition error ignoring speakers and line order: the best match
    among single hypothesis turns within +-margin seconds and their time-ordered concatenation."""
    out = []
    for t in ref:
        near = [h[3] for h in hyp if h[1] < t[2] + margin and t[1] - margin < h[2]]
        out.append(min([semi_global(t[3], c) for c in near + [''.join(near)]] + [len(t[3])]))
    return out


def start_offsets(ref, hyp):
    """Start offset of the most similar hypothesis turn (any speaker, within 15 s) for reference turns of
    >= 10 units whose best match has normalized indel similarity >= 0.7."""
    out = []
    for t in ref:
        if len(t[3]) < 10:
            continue
        cands = [(Indel.normalized_similarity(t[3], h[3]), h) for h in hyp if abs(h[1] - t[1]) < 15]
        if cands and max(c[0] for c in cands) >= 0.7:
            out.append(max(cands, key=lambda c: c[0])[1][1] - t[1])
    return out


def turn_diagnostics(predictions, label, similarity=None):
    """Turn-level diagnostics of reference turns (>3 units).

    concat (farfield-gap-diagnosis-20261013, kept for history): text error of turns that overlap another
    speaker vs those that do not, aligning the time-ordered concatenations of all speakers.
    speaker-aligned (model-compare-review-20261014, cpER alignment): the same split, plus the text-only lower
    bound (text_errors) and attribution = speaker-aligned minus text-only; attribution is also reported on
    the windows in the top tercile of voice similarity (``similarity`` file: window id -> highest cosine
    similarity between two reference speakers), where the gap to MOSS was largest.
    Also the share of matched turns whose start is off by more than 1 s, and windows whose output repeats
    lines across speakers (>= 3 pairs)."""
    kinds_all = ('overlapped', 'clean')
    units = dict.fromkeys(kinds_all, 0)
    concat, spk, txt = dict.fromkeys(kinds_all, 0), dict.fromkeys(kinds_all, 0), dict.fromkeys(kinds_all, 0)
    sims = json.loads(Path(similarity).read_text()) if similarity else {}
    cut = sorted(sims.values())[int(len(sims) * 2 / 3)] if sims else None
    high = {'units': 0, 'attribution': 0}
    offsets, duplicated, windows = [], 0, 0
    for line in Path(predictions).read_text().splitlines():
        row = json.loads(line)
        if row.get('label') != label:
            continue
        windows += 1
        ref, hyp = turns_of(row['reference']), turns_of(row.get('prediction'))
        duplicated += duplicate_pairs(hyp) >= 3
        offsets += start_offsets(ref, hyp)
        kinds = ['overlapped' if any(o[0] != t[0] and o[1] < t[2] and t[1] < o[2] for o in ref) else 'clean'
                 for t in ref]
        owner = [k for k, t in enumerate(ref) for _ in t[3]]
        for k in align_owners(''.join(t[3] for t in ref), ''.join(t[3] for t in hyp), owner):
            if k is not None and len(ref[k][3]) > 3:
                concat[kinds[k]] += 1
        se, te = speaker_errors(ref, hyp), text_errors(ref, hyp)
        is_high = cut is not None and sims.get(row['id'], -1) >= cut
        for k, t in enumerate(ref):
            if len(t[3]) > 3:
                units[kinds[k]] += len(t[3])
                spk[kinds[k]] += se[k]
                txt[kinds[k]] += te[k]
                if is_high:
                    high['units'] += len(t[3])
                    high['attribution'] += se[k] - te[k]
    rate = lambda a, b: a / max(b, 1)
    total = sum(units.values())
    out = {f'{kind}_turn_error_rate': rate(concat[kind], units[kind]) for kind in kinds_all}
    out |= {f'{kind}_turn_error_rate_speaker_aligned': rate(spk[kind], units[kind]) for kind in kinds_all}
    out |= {f'{kind}_turn_text_error_rate': rate(txt[kind], units[kind]) for kind in kinds_all}
    out['attribution_error_rate'] = rate(sum(spk.values()) - sum(txt.values()), total)
    if sims:
        out['high_similarity_attribution_error_rate'] = rate(high['attribution'], high['units'])
    out['start_offset_over_1s_share'] = rate(sum(abs(d) > 1 for d in offsets), len(offsets))
    return out | {'duplicate_speaker_windows': duplicated, 'windows': windows}


def evaluate_model(settings, model, step, gpu, port, prefix):
    """Serve one model, run every panel against it, return each panel's summary."""
    folder = Path(settings['output']) / 'checkpoints' / f'step-{step}'
    name, url = f'step{step}', f'http://127.0.0.1:{port}'
    serve = recipe(settings['serve_template'], folder, folder / 'preview')
    serve['task']['model'] = str(model)
    serve['task']['arguments'].update({'--port': port, '--served-model-name': name})
    serve['runtime']['gpus'] = [gpu]
    serve['recording'] = {'attempts_dir': {'path': str(folder / 'serve/attempts')}}
    (folder / 'serve.yaml').write_text(yaml.safe_dump(serve, sort_keys=False, allow_unicode=True))
    # vLLM listens with SO_REUSEPORT, so a second server on a busy port starts "successfully" and
    # the kernel splits requests between them; refuse a port that already has a listener.
    with socket.socket() as probe:
        if probe.connect_ex(('127.0.0.1', port)) == 0:
            raise RuntimeError(f'Port {port} already has a listener; give each selector its own port_base')
    with (folder / 'serve.log').open('a') as log:
        server = subprocess.Popen([sys.executable, '-m', 'open_audio_llm.cli', 'serve', '--config',
                                   str(folder / 'serve.yaml')], stdout=log, stderr=subprocess.STDOUT,
                                  start_new_session=True)
    try:
        ready(url, name, server, settings['server_startup_timeout'])
        jobs = {}
        for panel, spec in settings['panels'].items():
            config = recipe(spec['template'], folder, folder / 'preview')
            config['task']['server_url'] = url
            # Loading resolved {attempt} against the preview; each eval execution gets its own again.
            summary = {'path': '{attempt}/artifacts/evaluation/summary.json'}
            config['task']['arguments'].update({'--model': str(model), '--server-url': url, '--served-model': name,
                                                '--output_dir': {'path': '{attempt}/artifacts/evaluation'}})
            config['recording'] = {'attempts_dir': {'path': str(folder / panel / 'attempts')}, 'result': summary}
            config['tracking'].update(name=f'{prefix}-{panel}-step{step}',
                                      metrics=[{'file': summary, 'prefix': 'evaluation'}])
            path = folder / f'{panel}.yaml'
            path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True))
            log = (folder / f'{panel}.log').open('a')
            jobs[panel] = (subprocess.Popen([sys.executable, '-m', 'open_audio_llm.cli', 'eval', '--config',
                                             str(path)], stdout=log, stderr=subprocess.STDOUT), log)
        summaries = {}
        for panel, (process, log) in jobs.items():
            code = process.wait()
            log.close()
            if code:
                raise RuntimeError(f'{panel} evaluation of step {step} exited {code}')
            attempt = max((folder / panel / 'attempts').glob('[0-9]*'))
            summaries[panel] = json.loads((attempt / 'artifacts/evaluation/summary.json').read_text())
            for label in settings.get('turn_diagnostics', {}).get(panel, []):
                summaries[panel].setdefault('turn_diagnostics', {})[label] = turn_diagnostics(
                    attempt / 'artifacts/evaluation/predictions.jsonl', label,
                    settings.get('voice_similarity', {}).get(label))
            if settings.get('delete_request_audio'):
                # Per-request WAV copies of the panel audio; the panel itself stays in the dataset.
                subprocess.run(['rm', '-rf', str(attempt / 'artifacts/evaluation/audio')], check=True)
        return summaries
    finally:
        if server.poll() is None:
            os.killpg(server.pid, signal.SIGTERM)
            server.wait(timeout=300)


def metric(summaries, spec):
    if 'source' in spec:
        # KeSpeech reports one entry per dialect: pool them (errors over reference units).
        parts = [v for k, v in summaries[spec['panel']]['metrics'].items() if k.startswith(spec['source'] + '/')]
        assert parts, spec['source']
        return sum(p['errors'] for p in parts) / sum(p['reference_units'] for p in parts)
    if 'label' in spec:
        # Per-source value, e.g. cp_error_rate_without_loops: a single looping sample adds thousands
        # of errors, so the loop-inclusive rate moved 3-5 points between reruns of one checkpoint.
        return summaries[spec['panel']]['labels'][spec['label']][spec['field']]
    group = summaries[spec['panel']]['groups'][spec['group']]
    return group['cp_error_rate']


def record(settings, summaries, step, checkpoint):
    # Single-speaker ASR summaries carry per-source metrics only (no meeting loops or speaker counts).
    meetings = {panel: s for panel, s in summaries.items() if 'all' in s}
    loops = sum(s['all']['repetition_loops'] for s in meetings.values())
    row = {'step': step, 'checkpoint': str(checkpoint), 'loops': loops,
           'retention': {name: metric(summaries, spec) for name, spec in settings['retention'].items()},
           'target': metric(summaries, settings['target'])}
    for panel, summary in summaries.items():
        if panel not in meetings:
            for key, values in summary['metrics'].items():
                row[f'{panel}/{key}/error_rate'] = values['error_rate']
            continue
        for label, values in summary.get('labels', {}).items():
            row[f'{panel}/{label}/cp_error_rate_without_loops'] = values['cp_error_rate_without_loops']
            row[f'{panel}/{label}/repetition_loops'] = values['repetition_loops']
        for group, values in summary.get('groups', {}).items():
            row[f'{panel}/{group}/cp_error_rate'] = values['cp_error_rate']
            row[f'{panel}/{group}/repetition_loops'] = values['repetition_loops']
        for label, values in summary.get('turn_diagnostics', {}).items():
            for key, value in values.items():
                row[f'{panel}/{label}/{key}'] = value
        row[f'{panel}/speaker_count_accuracy'] = summary['all']['speaker_count_accuracy']
        row[f'{panel}/der'] = summary['all']['der']
    return row


def decide(settings, baseline, history):
    retention, loops = settings['retention'], settings['loops']
    ok_retention = lambda r: all(r['retention'][name] <= baseline['retention'][name] + spec['tolerance']  # noqa: E731
                                 for name, spec in retention.items())
    ok_loops = lambda r: r['loops'] <= baseline['loops'] + loops['max_increase']  # noqa: E731
    patience = settings['patience']
    recent = history[-patience:]
    reason = None
    if len(recent) == patience and not any(ok_retention(r) for r in recent):
        reason = 'consecutive_retention_regression'
    elif loops.get('stop', True) and len(recent) == patience and not any(ok_loops(r) for r in recent):
        reason = 'consecutive_repetition_loop_increase'
    eligible = [r for r in history if ok_retention(r) and ok_loops(r)]
    best = min(eligible, key=lambda r: r['target']) if eligible else None
    return {'step': history[-1]['step'] if history else 0, 'stop': reason is not None, 'reason': reason,
            'best_checkpoint': best['checkpoint'] if best else None, 'best_step': best['step'] if best else None,
            'best_target': best['target'] if best else None, 'baseline': baseline,
            'improved': best is not None and best['target'] < baseline['target'],
            'evaluated_steps': [r['step'] for r in history]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    settings = yaml.safe_load(parser.parse_args().config.read_text())['parameters']
    state = Path(settings['output'])
    state.mkdir(parents=True, exist_ok=True)
    prefix = settings['tracking_prefix']
    gpus = list(settings['gpus'])
    free = {gpu: settings['port_base'] + i for i, gpu in enumerate(gpus)}
    baseline_file = state / 'baseline.json'
    history_file = state / 'selection-history.json'
    if settings.get('previous_state') and not history_file.exists():
        # Continue a stopped selection: its baseline and evaluated steps stay valid; weights are fetched again.
        for name in ('baseline.json', 'selection-history.json'):
            shutil.copy2(Path(settings['previous_state']) / name, state / name)
    if settings.get('baseline') and not baseline_file.exists():
        # Measured with the same chain before this experiment; see the README for its execution.
        shutil.copy2(settings['baseline'], baseline_file)
    if not baseline_file.exists():
        gpu = gpus[0]
        summaries = evaluate_model(settings, settings['base_model'], 0, gpu, free[gpu], prefix)
        baseline_file.write_text(json.dumps(record(settings, summaries, 0, settings['base_model']), indent=2) + '\n')
    baseline = json.loads(baseline_file.read_text())
    history_file = state / 'selection-history.json'
    history = json.loads(history_file.read_text()) if history_file.exists() else []

    def write_decision():
        ordered = sorted(history, key=lambda r: r['step'])
        decision = decide(settings, baseline, ordered)
        history_file.write_text(json.dumps(ordered, indent=2) + '\n')
        (state / 'selection-decision.json').write_text(json.dumps(decision, indent=2) + '\n')
        return decision

    decision = write_decision()
    attempts = [Path(p) for p in settings['training_attempts']]
    if not attempts:
        return
    running = attempts[-1]
    started = set()

    def run(step, checkpoint):
        with LOCK:
            gpu = next(g for g, port in free.items() if port is not None)
            port, free[gpu] = free[gpu], None
        try:
            model = materialize(settings, checkpoint, step)
            summaries = evaluate_model(settings, model, step, gpu, port, prefix)
        finally:
            with LOCK:
                free[gpu] = port
        with LOCK:
            history.append(record(settings, summaries, step, model))
            decision = write_decision()
            # The training directory belongs to root on ACP; the control directory is writable here.
            stop = Path(settings['stop_request'])
            if decision['stop'] and not stop.exists():
                stop.write_text(json.dumps({'step': step, 'reason': decision['reason']}, indent=2) + '\n')

    with ThreadPoolExecutor(len(gpus)) as pool:
        futures = []
        while True:
            finished = json.loads((running / 'status.json').read_text())['execution'] in {
                'completed', 'failed', 'interrupted'}
            done = {r['step'] for r in history} | started
            pending = [(s, p) for a in attempts
                       for s, p in complete_checkpoints(a / 'artifacts/training', settings['interval'],
                                                        settings['max_steps']) if s not in done]
            for step, checkpoint in pending:
                started.add(step)
                futures.append(pool.submit(run, step, checkpoint))
            for future in [f for f in futures if f.done()]:
                future.result()
            if finished and not pending and all(f.done() for f in futures):
                break
            time.sleep(settings['poll_seconds'])
        for future in futures:
            future.result()
    decision = write_decision()
    if json.loads((running / 'status.json').read_text())['execution'] != 'completed' and not decision['stop']:
        raise SystemExit(f'Training attempt did not complete: {running}')
    chosen, chosen_step = decision['best_checkpoint'], decision['best_step']
    if not chosen and history:
        # No candidate kept retention and loops: still report the stage with its last checkpoint.
        last = max(history, key=lambda r: r['step'])
        chosen, chosen_step = last['checkpoint'], last['step']
    result = {'selection_passed': decision['improved'], 'checkpoint': chosen, 'decision': decision}
    if chosen and settings.get('final_panels'):
        final = {**settings, 'panels': settings['final_panels'], 'output': str(state / 'final')}
        gpu = gpus[0]
        result['final'] = evaluate_model(final, chosen, chosen_step, gpu, free[gpu], prefix + '-final')
        baselines = {panel: json.loads(Path(path).read_text())
                     for panel, path in (settings.get('final_baseline') or {}).items()}
        missing = {p: s for p, s in settings['final_panels'].items() if p not in baselines}
        if missing:
            # Panels without an earlier same-chain baseline are measured on the start model now.
            start = {**final, 'panels': missing, 'output': str(state / 'final-baseline')}
            baselines.update(evaluate_model(start, settings['base_model'], 0, gpu, free[gpu], prefix + '-final-base'))
        checks = {}
        for panel, before in baselines.items():
            after = result['final'][panel]
            checks[f'{panel}/loops_not_increased'] = (
                after['all']['repetition_loops'] <= before['all']['repetition_loops'])
            for group, values in after.get('groups', {}).items():
                delta = values['cp_error_rate'] - before['groups'][group]['cp_error_rate']
                result.setdefault('final_delta', {})[f'{panel}/{group}'] = delta
                checks[f'{panel}/{group}/not_worse'] = delta <= settings['final_tolerance']
        result['checks'] = checks
        result['passed'] = decision['improved'] and all(checks.values())
    (state / 'final-results.json').write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()
