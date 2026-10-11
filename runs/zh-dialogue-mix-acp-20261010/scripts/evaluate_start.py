"""Evaluate the start checkpoint on panels the selector only measures at the end.

The selector compares the final checkpoint with the chain's first model; this
records the direct start point of this round (the previous round's selected
checkpoint) with the same serve and panel templates.
"""
import argparse
import json
from pathlib import Path

import yaml

from select_checkpoints import evaluate_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    settings = yaml.safe_load(parser.parse_args().config.read_text())['parameters']
    summaries = evaluate_model(settings, settings['model'], settings['step'], settings['gpu'],
                               settings['port'], settings['tracking_prefix'])
    Path(settings['output'], 'start-results.json').write_text(
        json.dumps({'model': settings['model'], 'summaries': summaries}, indent=2, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()
