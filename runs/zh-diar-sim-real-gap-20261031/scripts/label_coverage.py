"""How much audible activity lies outside the labeled turns (energy check of label coverage).

Per window: 20 ms frame log energy; a frame is active when it is 15 dB above the window's 10th-percentile frame
(its noise floor). Reported: share of active frames farther than ``pad`` s from every labeled turn. Unlabeled
backchannels, laughter or side talk show up here; a source whose labels drop short turns has a high share."""
import argparse
import gzip
import json
import random
import re
from pathlib import Path

import numpy as np
import soundfile as sf
from audio_data_contract import load_catalog, load_roots, resolve_split

TURN = re.compile(r'^\[(S\d+)\]\[(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?)\]')
HOP = 320


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--catalog', type=Path, required=True)
    p.add_argument('--roots', type=Path, required=True)
    p.add_argument('--source', action='append', required=True, help='name=dataset@version:split')
    p.add_argument('--sample', type=int, default=60)
    p.add_argument('--pad', type=float, default=0.3)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    catalog, roots = load_catalog(args.catalog), load_roots(args.roots)
    out = {}
    for item in args.source:
        name, ref = item.split('=', 1)
        did, rest = ref.split('@', 1)
        ver, split = rest.split(':', 1)
        index = {}
        for path in resolve_split(catalog, did, ver, split, 'audio_index', roots):
            with gzip.open(path, 'rt') as f:
                for line in f:
                    row = json.loads(line)
                    index[row['cut_id']] = row
        records = []
        for path in resolve_split(catalog, did, ver, split, 'records', roots):
            with gzip.open(path, 'rt') as f:
                records += [json.loads(line) for line in f]
        random.Random(20261031).shuffle(records)
        shares, outside_s, minutes = [], 0.0, 0.0
        for r in records[:args.sample]:
            a = r['audio_slots'][0]['ref']
            row = index[a['cut_id']]
            path = Path(roots[row['root_alias']]) / row['relative_path']
            info = sf.info(path)
            start = int(round(a.get('start', 0.0) * info.samplerate))
            frames = int(round(a['duration'] * info.samplerate))
            audio, rate = sf.read(path, start=start, frames=frames, dtype='float32', always_2d=True)
            ch = a.get('channel')
            x = audio[:, ch] if isinstance(ch, int) else audio.mean(axis=1)
            n = len(x) // HOP
            e = 10 * np.log10((x[:n * HOP].reshape(n, HOP) ** 2).mean(1) + 1e-10)
            active = e > np.percentile(e, 10) + 15
            labeled = np.zeros(n, dtype=bool)
            for line in (r.get('target') or '').splitlines():
                m = TURN.match(line.strip())
                if m:
                    s, t = float(m[2]) - args.pad, float(m[3]) + args.pad
                    labeled[max(0, int(s * rate / HOP)):min(n, int(t * rate / HOP) + 1)] = True
            if active.sum() == 0:
                continue
            shares.append(float((active & ~labeled).sum() / active.sum()))
            outside_s += (active & ~labeled).sum() * HOP / rate
            minutes += n * HOP / rate / 60
        out[name] = {'windows': len(shares), 'unlabeled_active_share_mean': float(np.mean(shares)),
                     'unlabeled_active_share_median': float(np.median(shares)),
                     'unlabeled_active_seconds_per_min': outside_s / max(minutes, 1e-6)}
        print(name, json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in out[name].items()}), flush=True)
    args.output.write_text(json.dumps(out, indent=2) + '\n')


if __name__ == '__main__':
    main()
