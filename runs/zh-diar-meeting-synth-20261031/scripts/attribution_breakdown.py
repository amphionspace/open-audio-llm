"""Where do speaker-attribution errors come from, per variant? (existing decodes, no new inference)

For every reference turn of > 3 units the attribution overhead is the edits charged under the cpER speaker mapping
minus the speaker-free text errors. Overhead goes to the speaker that actually holds the turn's text (best semi-global
match within the text margin):
  split    a hypothesis speaker no reference speaker is mapped to (one voice heard as two)
  swap     a hypothesis speaker mapped to another reference speaker
  merged   the reference speaker has no hypothesis speaker at all (two voices heard as one)
  same     the mapped speaker itself holds it (line order or deletion, not attribution)
  lost     no hypothesis turn matches it within MATCH"""
import argparse
import collections
import json
from pathlib import Path

import numpy as np

import overlap_breakdown as ob
import select_checkpoints as sc

MATCH = 0.34


def window_stats(ref, hyp):
    mapping, _, _ = sc.speaker_mapping(ref, hyp)
    te, se = sc.text_errors(ref, hyp), sc.speaker_errors(ref, hyp)
    reverse = {h: r for r, h in mapping.items()}
    out = []
    for k, t in enumerate(ref):
        if len(t[3]) <= 3:
            continue
        over = max(se[k] - te[k], 0)
        kind = None
        if over:
            near = [h for h in hyp if h[1] < t[2] + sc.TEXT_MARGIN and t[1] - sc.TEXT_MARGIN < h[2] and h[3]]
            best = min(near, key=lambda h: sc.semi_global(t[3], h[3]), default=None)
            if best is None or sc.semi_global(t[3], best[3]) / len(t[3]) > MATCH:
                kind = 'lost'
            elif t[0] not in mapping:
                kind = 'merged'
            elif best[0] == mapping[t[0]]:
                kind = 'same'
            elif best[0] in reverse:
                kind = 'swap'
            else:
                kind = 'split'
        out.append((ob.category(t, ref), len(t[3]), te[k], se[k], over, kind))
    return out, len({t[0] for t in ref}), len({h[0] for h in hyp})


def analyze(files, label):
    per_window, units, text, spk = collections.defaultdict(lambda: [0, 0]), collections.Counter(), collections.Counter(), collections.Counter()
    kinds, counts = collections.Counter(), collections.Counter()
    decodes = 0
    for f in files:
        for line in Path(f).read_text().splitlines():
            row = json.loads(line)
            if row.get('label') != label:
                continue
            ref, hyp = sc.turns_of(row['reference']), sc.turns_of(row.get('prediction'))
            stats, nr, nh = window_stats(ref, hyp)
            decodes += 1
            counts['exact' if nh == nr else ('over' if nh > nr else 'under')] += 1
            counts['extra_speakers'] += max(nh - nr, 0)
            counts['missing_speakers'] += max(nr - nh, 0)
            for c, n, te, se, over, kind in stats:
                units[c] += n; text[c] += te; spk[c] += se
                if c == 'none':
                    per_window[row['id']][0] += over; per_window[row['id']][1] += n
                    if kind:
                        kinds[kind] += over
    none = units['none']
    rates = sorted((v[0] for v in per_window.values()), reverse=True)
    top = sum(rates[:max(len(rates) // 10, 1)]) / max(sum(rates), 1)
    return {'decodes': decodes, 'windows': len(per_window),
            'no_overlap': {'unit_share': round(none / max(sum(units.values()), 1), 3),
                           'text_pct': round(100 * text['none'] / max(none, 1), 2),
                           'overhead_pct': round(100 * (spk['none'] - text['none']) / max(none, 1), 2),
                           'overhead_pct_by_kind': {k: round(100 * v / max(none, 1), 2) for k, v in kinds.most_common()},
                           'top10pct_windows_share_of_overhead': round(top, 3),
                           'windows_with_no_overhead': round(float(np.mean([r == 0 for r in rates])), 3)},
            'overhead_pct_by_category': {c: round(100 * (spk[c] - text[c]) / max(units[c], 1), 2) for c in ob.CATS},
            'speaker_count': {'exact': round(counts['exact'] / max(decodes, 1), 3), 'over': round(counts['over'] / max(decodes, 1), 3),
                              'under': round(counts['under'] / max(decodes, 1), 3),
                              'extra_per_decode': round(counts['extra_speakers'] / max(decodes, 1), 3),
                              'missing_per_decode': round(counts['missing_speakers'] / max(decodes, 1), 3)}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--item', action='append', required=True, help='name=label:pred1,pred2,...')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    out = {}
    for item in args.item:
        name, rest = item.split('=', 1)
        label, files = rest.split(':', 1)
        out[name] = analyze(files.split(','), label)
        print(name, json.dumps(out[name], ensure_ascii=False), flush=True)
    args.output.write_text(json.dumps(out, ensure_ascii=False, indent=1) + '\n')


if __name__ == '__main__':
    main()
