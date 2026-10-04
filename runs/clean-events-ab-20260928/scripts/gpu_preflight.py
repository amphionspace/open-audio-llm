"""Preserve the pilot's memory guard after the historical busy-GPU OOM."""

import json
import subprocess

from settings import SETTINGS, OUTPUT

rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used,memory.free',
                                '--format=csv,noheader,nounits'], text=True)
gpus = [[int(field.strip()) for field in row.split(',')] for row in rows.strip().splitlines()]
selected = [row for row in gpus if row[0] in SETTINGS['runtime']['gpus']]
# 1024 MiB is the original experiment's preexisting-memory ceiling.
assert len(selected) == len(SETTINGS['runtime']['gpus']) and all(row[1] < 1024 for row in selected), selected
(OUTPUT / 'gpu-preflight.json').write_text(json.dumps({'gpus': selected}, indent=2) + '\n')
