"""Text / speaker-aligned error of reference turns by how much of each turn other speakers cover.

Categories (turns of > 3 units): no overlap; overlapped only by others' turns of <= 2 units (backchannels); covered
< 30%, 30-70%, >= 70% of the turn's duration. Several prediction files (repeated decodes, or one per variant) can be
pooled; labels select the rows."""
import argparse
import collections
import json
from pathlib import Path

import select_checkpoints as sc

CATS = ('none', 'backchannel_only', 'light_lt30', 'medium_30_70', 'heavy_ge70')


def covered(t, others, step=0.05):
    n = int((t[2] - t[1]) / step) + 1
    return sum(any(o[1] <= t[1] + k * step < o[2] for o in others) for k in range(n)) / n


def category(t, ref):
    others = [o for o in ref if o[0] != t[0] and o[1] < t[2] and t[1] < o[2]]
    if not others:
        return 'none'
    if all(len(o[3]) <= 2 for o in others):
        return 'backchannel_only'
    f = covered(t, others)
    return 'light_lt30' if f < 0.3 else ('medium_30_70' if f < 0.7 else 'heavy_ge70')


def breakdown(files, label):
    units, txt, spk = collections.Counter(), collections.Counter(), collections.Counter()
    for f in files:
        for line in Path(f).read_text().splitlines():
            row = json.loads(line)
            if row.get('label') != label:
                continue
            ref, hyp = sc.turns_of(row['reference']), sc.turns_of(row.get('prediction'))
            te, se = sc.text_errors(ref, hyp), sc.speaker_errors(ref, hyp)
            for k, t in enumerate(ref):
                if len(t[3]) > 3:
                    c = category(t, ref)
                    units[c] += len(t[3]); txt[c] += te[k]; spk[c] += se[k]
    total = sum(units.values())
    return {c: {'unit_share': units[c] / total, 'text_error_rate': txt[c] / max(units[c], 1),
                'speaker_aligned_error_rate': spk[c] / max(units[c], 1)} for c in CATS}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--item', action='append', required=True, help='name=label:pred1,pred2,...')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    out = {}
    for item in args.item:
        name, rest = item.split('=', 1)
        label, files = rest.split(':', 1)
        out[name] = breakdown(files.split(','), label)
        print(name, ' '.join(f"{c}:{v['text_error_rate']*100:.1f}" for c, v in out[name].items()), flush=True)
    args.output.write_text(json.dumps(out, indent=2) + '\n')


if __name__ == '__main__':
    main()
