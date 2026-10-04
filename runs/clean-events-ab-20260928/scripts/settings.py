"""Explicit experiment locations, loaded from the launcher's effective YAML."""

import argparse
from pathlib import Path
import sys

import yaml

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--config', type=Path, required=True)
options, remaining = parser.parse_known_args()
sys.argv = [sys.argv[0], *remaining]
CONFIG_FILE = options.config.absolute()
SETTINGS = yaml.safe_load(CONFIG_FILE.read_text())
ROOT = Path(SETTINGS['experiment']['root'])
RUNS = Path(SETTINGS['experiment']['runs'])
OUTPUT = Path(SETTINGS['experiment']['output'])


def artifact(name):
    name = str(name)
    parts = Path(name).parts
    mapped = SETTINGS['experiment']['files'].get(parts[0]) if parts else None
    return Path(mapped).joinpath(*parts[1:]) if mapped else OUTPUT / name


def child(script, *args, python=None):
    return [python or SETTINGS['runtime']['python'], str(artifact(script)),
            '--config', str(CONFIG_FILE), *map(str, args)]
