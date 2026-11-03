"""Re-decide finished selections (rounds 1 and 2) with the smoothed paired rule of select_checkpoints.decide.

Per-window scores come from each evaluated step's selection panel (meeting-scores.jsonl); the baseline is the start
model's step-0 measurement of round 1 arm A, which every arm of both rounds used."""
import argparse
import json
from pathlib import Path

import yaml

import select_checkpoints as sc


def windows_of(evaluation):
    return {r['id']: [r['cp_errors'], r['units'], bool(r.get('repetition_loop'))]
            for r in map(json.loads, (evaluation / 'meeting-scores.jsonl').read_text().splitlines())
            if r.get('label') == 'alimeeting_dev'}


def step_eval(task, step):
    hits = sorted(task.glob(f'attempts/*/artifacts/checkpoints/step-{step}/selection/attempts/*/artifacts/evaluation'))
    hits = [h for h in hits if (h / 'meeting-scores.jsonl').exists()]
    return hits[-1] if hits else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--select-config', type=Path, required=True, help='round-2 select config (retention/loop gates)')
    p.add_argument('--baseline-eval', type=Path, required=True)
    p.add_argument('--arm', action='append', required=True, help='name=path/to/tasks/03-select-x')
    p.add_argument('--smoothing', type=int, default=3)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    settings = yaml.safe_load(args.select_config.read_text())['parameters']
    settings['smoothing'] = args.smoothing
    out = {}
    for item in args.arm:
        name, path = item.split('=', 1)
        task = Path(path)
        history_file = sorted(task.glob('attempts/*/artifacts/selection-history.json'))[-1]
        history = sorted(json.loads(history_file.read_text()), key=lambda r: r['step'])
        baseline = json.loads((history_file.parent / 'baseline.json').read_text())
        baseline['target_windows'] = windows_of(args.baseline_eval)
        missing = []
        for r in history:
            e = step_eval(task, r['step'])
            if e is None:
                missing.append(r['step'])
            else:
                r['target_windows'] = windows_of(e)
        history = [r for r in history if r.get('target_windows')]
        old = sc.decide({**settings, 'smoothing': 0}, baseline, history)
        new = sc.decide(settings, baseline, history)
        out[name] = {'missing_steps': missing, 'old': {k: old[k] for k in ('best_step', 'best_target', 'improved')},
                     'new': new['smoothed']}
        s = new['smoothed']
        if s is None:
            print(f"{name}: 旧规则 最好单点 {old['best_step']} improved={old['improved']} | 新规则 没有连续 "
                  f"{args.smoothing} 个通过保持/复读门槛的保存点（已评 {[r['step'] for r in history]}）", flush=True)
            continue
        print(f"{name}: 旧规则 最好单点 {old['best_step']} {old['best_target']*100:.2f} improved={old['improved']} | "
              f"新规则 最好三连 {s['steps']} 合并 {s['candidate']*100:.2f}（同窗口起点 {s['baseline']*100:.2f}）"
              f" 差 {s['difference']*100:+.2f} [{s['interval_95'][0]*100:+.2f}, {s['interval_95'][1]*100:+.2f}]"
              f" improved={s['improved']} 窗口 {s['windows']}" + (f" 缺 {missing}" if missing else ''), flush=True)
    args.output.write_text(json.dumps(out, indent=2) + '\n')


if __name__ == '__main__':
    main()
