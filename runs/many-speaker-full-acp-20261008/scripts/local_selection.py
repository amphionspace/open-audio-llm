"""Select checkpoints on the development machine while ACP training continues.

Each saved checkpoint is evaluated with the unchanged selection panels. A stop
decision is written to the running attempt as stop-request.json, which the
training callback broadcasts to every rank. After training ends, the chosen
checkpoint runs the two fixed reporting panels.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import time

import yaml

sys.path.insert(0, str(Path(__file__).parent))
from checkpoint_eval import main as evaluate_checkpoint  # noqa: E402
from final_eval import main as final_evaluation  # noqa: E402


def complete_checkpoints(training, interval, max_steps):
    for path in sorted(training.glob('checkpoint-[0-9]*'), key=lambda p: int(p.name.split('-')[1])):
        step = int(path.name.split('-')[1])
        # The sampler state is written after the weights and optimizer, so it marks a finished save.
        # ACP writes weights as root with mode 0600; wait until they are readable here.
        if (step % interval == 0 or step == max_steps) and (path/'catalog_sampler.json').is_file() \
                and all(os.access(p, os.R_OK) for p in path.glob('model*.safetensors')):
            yield step, path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    config = yaml.safe_load(parser.parse_args().config.read_text())
    settings = config['parameters']
    state = Path(settings['output'])
    state.mkdir(parents=True, exist_ok=True)
    attempts = [Path(p) for p in settings['training_attempts']]
    trainings = [a/'artifacts/training' for a in attempts]
    running = attempts[-1]
    history_file = state/'selection-history.json'
    if not history_file.exists():
        # Evaluations already finished inside earlier attempts keep their results.
        history = []
        # Earlier attempts' in-job evaluations and an interrupted selector's finished
        # checkpoints keep their results instead of being evaluated again.
        sources = [Path(p) for p in settings.get('previous_selections', [])] + trainings[:-1]
        for source in sources:
            if (source/'selection-history.json').is_file():
                steps = {r['step'] for r in history}
                history += [r for r in json.loads((source/'selection-history.json').read_text())
                            if r['step'] not in steps]
            baseline = source/'selection-baseline/summary.json'
            # Only the summary is needed; the baseline attempt links to model weights.
            if baseline.is_file() and not (state/'selection-baseline/summary.json').exists():
                (state/'selection-baseline').mkdir(parents=True, exist_ok=True)
                shutil.copy2(baseline, state/'selection-baseline/summary.json')
        history_file.write_text(json.dumps(history, indent=2)+'\n')
        # The latest earlier decision covers that history until a new checkpoint is evaluated.
        for source in reversed(sources):
            if (source/'selection-decision.json').is_file():
                shutil.copy2(source/'selection-decision.json', state/'selection-decision.json')
                break
    while True:
        finished = json.loads((running/'status.json').read_text())['execution'] in {'completed', 'failed'}
        done = {r['step'] for r in json.loads(history_file.read_text())}
        pending = [(s, p) for t in trainings for s, p in complete_checkpoints(t, settings['interval'], settings['max_steps'])
                   if s not in done]
        for step, checkpoint in pending:
            evaluate_checkpoint(checkpoint, state/'checkpoints'/checkpoint.name, state)
            decision = json.loads((state/'selection-decision.json').read_text())
            if decision['stop'] and not (running/'artifacts/training/stop-request.json').exists():
                (running/'artifacts/training/stop-request.json').write_text(json.dumps(
                    {'step': step, 'reason': decision['reason']}, indent=2)+'\n')
        if finished and not pending:
            break
        time.sleep(settings['poll_seconds'])
    # An explicitly configured checkpoint reports a stage that ended without its last save.
    chosen = settings.get('final_checkpoint')
    if not chosen and json.loads((running/'status.json').read_text())['execution'] != 'completed':
        raise SystemExit(f'Training attempt did not complete: {running}')
    final_evaluation(state, running/'effective.yaml', chosen)


if __name__ == '__main__':
    main()
