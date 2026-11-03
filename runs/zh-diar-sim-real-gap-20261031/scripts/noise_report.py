"""Decode noise of one model on the selection panel: repeated evaluations compared window by window.

For every pair of runs: aggregate loop-free AliMeeting dev cpER of each, the sum of |per-window error change|,
the net change, and the share of windows whose output text is byte-identical."""
import argparse
import itertools
import json
from pathlib import Path


def load(evaluation):
    scores = {r['id']: r for r in map(json.loads, (evaluation / 'meeting-scores.jsonl').read_text().splitlines())
              if r.get('label') == 'alimeeting_dev'}
    preds = {r['id']: r.get('prediction') for r in map(json.loads, (evaluation / 'predictions.jsonl').read_text().splitlines())
             if r.get('label') == 'alimeeting_dev'}
    return scores, preds


def rate(scores, ids):
    keep = [i for i in ids if not scores[i].get('repetition_loop')]
    return sum(scores[i]['cp_errors'] for i in keep) / sum(scores[i]['units'] for i in keep)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', action='append', required=True, help='name=evaluation dir')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    runs = {k: load(Path(v)) for k, v in (x.split('=', 1) for x in args.run)}
    out = {'runs': {}, 'pairs': []}
    for name, (scores, _) in runs.items():
        ids = sorted(scores)
        out['runs'][name] = {'cpER_without_loops': rate(scores, ids),
                             'loops': sum(bool(s.get('repetition_loop')) for s in scores.values())}
        print(f"{name}: cpER(无复读) {out['runs'][name]['cpER_without_loops']*100:.2f}  复读窗 {out['runs'][name]['loops']}")
    for (a, (sa, pa)), (b, (sb, pb)) in itertools.combinations(runs.items(), 2):
        ids = sorted(set(sa) & set(sb))
        both = [i for i in ids if not sa[i].get('repetition_loop') and not sb[i].get('repetition_loop')]
        d = [sb[i]['cp_errors'] - sa[i]['cp_errors'] for i in both]
        units = sum(sa[i]['units'] for i in both)
        same = sum(pa.get(i) == pb.get(i) for i in ids)
        row = {'a': a, 'b': b, 'windows': len(both), 'abs_change_errors': sum(abs(x) for x in d),
               'net_change_points': 100 * sum(d) / units, 'identical_output_share': same / len(ids),
               'changed_windows': sum(x != 0 for x in d)}
        out['pairs'].append(row)
        print(f"{a} vs {b}: 输出完全相同 {row['identical_output_share']*100:.1f}%  变化窗 {row['changed_windows']}/{len(both)}"
              f"  逐窗口变化总量 {row['abs_change_errors']}  净变化 {row['net_change_points']:+.2f} 点")
    args.output.write_text(json.dumps(out, indent=2) + '\n')


if __name__ == '__main__':
    main()
