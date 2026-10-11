"""Evaluate every selected-run checkpoint on the dialogue panels while training runs.

The selection panels do not contain the new training sources' held-out sets;
this follows the selector's copies (models/checkpoint-<step> with complete.json)
and serves each one on its own GPU with the same serve and panel templates.
One row per checkpoint goes to history.jsonl.
"""
import argparse
import json
import time
from pathlib import Path

import yaml

from select_checkpoints import evaluate_model

FIELDS = ('cp_error_rate', 'cp_error_rate_without_loops', 'repetition_loops', 'speaker_count_accuracy', 'der')


def running(attempt):
    return json.loads((Path(attempt) / 'status.json').read_text())['execution'] in ('starting', 'submitted', 'running')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    settings = yaml.safe_load(parser.parse_args().config.read_text())['parameters']
    history = Path(settings['output']) / 'history.jsonl'
    done = {json.loads(line)['step'] for line in history.read_text().splitlines()} if history.exists() else set()
    for step in range(settings['interval'], settings['max_steps'] + 1, settings['interval']):
        if step in done:
            continue
        model = Path(settings['models']) / f'checkpoint-{step}'
        # The selector writes complete.json after a full copy; it stops copying once training and
        # selection have ended, so a checkpoint still missing then was never saved.
        while not (model / 'complete.json').is_file():
            if not running(settings['training_attempt']) and not running(settings['selection_attempt']):
                return
            time.sleep(settings['poll_seconds'])
        summaries = evaluate_model(settings, model, step, settings['gpu'], settings['port'],
                                   settings['tracking_prefix'])
        row = {'step': step, **{panel: {k: s['all'][k] for k in FIELDS} for panel, s in summaries.items()}}
        with history.open('a') as stream:
            stream.write(json.dumps(row) + '\n')


if __name__ == '__main__':
    main()
