"""Keep the supplemental recovery within the synthesis task's execution record."""

import json
import subprocess

from settings import artifact, child

initial = subprocess.run(child('generate.py'))
if initial.returncode:
    gate = artifact('release-gate.json')
    if not gate.exists() or json.loads(gate.read_text())['passed']:
        raise SystemExit(initial.returncode)
    # The historical recovery keeps accepted mixtures and appends the same English supplement.
    subprocess.run(child('supplement.py'), check=True)
