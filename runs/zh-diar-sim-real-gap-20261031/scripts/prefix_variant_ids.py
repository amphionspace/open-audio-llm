"""Make variant record ids unique for AmphionEval ("<variant>/<panel id>") and write the per-variant voice-similarity
map; run once after build_dev_variants.py (AmphionEval rejects duplicate sample ids across splits)."""
import gzip
import hashlib
import json
import sys
from pathlib import Path

import yaml

catalog_dir, run_dir = Path(sys.argv[1]), Path(sys.argv[2])
cat = yaml.safe_load((catalog_dir / 'catalog.yaml').read_text())
entry = cat[0]
for art in entry['artifacts']:
    if art['kind'] != 'audio-records':
        continue
    path = catalog_dir / art['relative_path']
    variant = art['name'].removesuffix('-records')
    rows = [json.loads(line) for line in gzip.open(path, 'rt')]
    for r in rows:
        if not r['id'].startswith(variant + '/'):
            r['id'] = f"{variant}/{r['id']}"
    with gzip.open(path, 'wt') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    art['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
(catalog_dir / 'catalog.yaml').write_text(yaml.safe_dump(cat, sort_keys=False, allow_unicode=True))
sims = json.loads((run_dir / 'shared/alimeeting-dev-voice-similarity.json').read_text())
variants = [a['name'].removesuffix('-records') for a in entry['artifacts'] if a['kind'] == 'audio-records']
(run_dir / 'shared/voice-similarity-variants.json').write_text(
    json.dumps({f'{v}/{k}': s for v in variants for k, s in sims.items()}, indent=1) + '\n')
