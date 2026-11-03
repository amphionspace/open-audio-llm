"""Per-turn text error of the same dev turns heard in overlap (near_gated) and laid out without overlap (near_seq),
grouped by each turn's overlap category in the original meeting (overlap_breakdown.category)."""
import argparse
import collections
import json
from pathlib import Path

import select_checkpoints as sc
from overlap_breakdown import CATS, category


def rows(path, label):
    return {r['id'].split('/', 1)[1]: r for r in map(json.loads, Path(path).read_text().splitlines())
            if r.get('label') == label}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--predictions', type=Path, required=True, help='predictions.jsonl holding both labels')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    over, seq = rows(args.predictions, 'near_gated'), rows(args.predictions, 'near_seq')
    units, e_over, e_seq = collections.Counter(), collections.Counter(), collections.Counter()
    for wid in sorted(set(over) & set(seq)):
        ref_o, hyp_o = sc.turns_of(over[wid]['reference']), sc.turns_of(over[wid].get('prediction'))
        ref_s, hyp_s = sc.turns_of(seq[wid]['reference']), sc.turns_of(seq[wid].get('prediction'))
        te_o, te_s = sc.text_errors(ref_o, hyp_o), sc.text_errors(ref_s, hyp_s)
        # Same turns in the same start order; match by (speaker, text) occurrence to be safe.
        slot = collections.defaultdict(list)
        for k, t in enumerate(ref_s):
            slot[(t[0], t[3])].append(k)
        for k, t in enumerate(ref_o):
            if len(t[3]) <= 3 or not slot[(t[0], t[3])]:
                continue
            j = slot[(t[0], t[3])].pop(0)
            c = category(t, ref_o)
            units[c] += len(t[3]); e_over[c] += te_o[k]; e_seq[c] += te_s[j]
    out = {c: {'unit_share': units[c] / sum(units.values()), 'in_overlap': e_over[c] / max(units[c], 1),
               'sequential': e_seq[c] / max(units[c], 1)} for c in CATS}
    for c, v in out.items():
        print(f"{c:18s} 字数占比 {v['unit_share']*100:5.1f}%  重叠中 {v['in_overlap']*100:5.1f}%  排开后 {v['sequential']*100:5.1f}%")
    args.output.write_text(json.dumps(out, indent=2) + '\n')


if __name__ == '__main__':
    main()
