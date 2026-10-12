"""Summarize the decode-setting study from the per-sample AmphionEval scores of every job.

- Per (model, setting, repeat, split): AmphionEval's own summary (cpER, loop-excluded cpER, loops,
  DER, speaker-count accuracy).
- Common-subset cpER: per model and split, only samples that loop in none of that model's runs, so
  settings and repeats are compared on identical samples (the loop-excluded cpER of each run drops a
  different set).
- Paired loops: per model, which samples loop under which setting/repeat, and per setting the loops
  fixed and introduced relative to default r1.
- Rerun spread: per model with several default repeats, range and sample std of loops and the range
  of loop-excluded and common-subset cpER.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics

import yaml

FIELDS = ('samples', 'cp_error_rate', 'cp_error_rate_without_loops', 'repetition_loops', 'der',
          'speaker_count_accuracy')


def rate(rows):
    units = sum(r['units'] for r in rows)
    return sum(r['cp_errors'] for r in rows) / units if units else None


def load(jobs_dir):
    runs = {}
    for path in sorted(jobs_dir.glob('*/*/r*/result.json')):
        result = json.loads(path.read_text())
        scores = {}
        for panel, values in result['panels'].items():
            # Rescored panels (see rescore.py) carry their own scores file.
            path = values.get('scores_file') or Path(values['attempt']) / 'artifacts/evaluation/meeting-scores.jsonl'
            lines = Path(path).read_text(encoding='utf-8').split('\n')
            scores[panel] = [json.loads(line) for line in lines if line.strip()]
        runs[result['key']] = {**result, 'scores': scores}
    return runs


def pct(value):
    return '—' if value is None else f'{100 * value:.2f}'


def spread(values):
    values = [v for v in values if v is not None]
    return {'values': values, 'min': min(values), 'max': max(values), 'range': max(values) - min(values),
            'mean': statistics.mean(values), 'std': statistics.stdev(values) if len(values) > 1 else 0.0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    settings = yaml.safe_load(parser.parse_args().config.read_text())['parameters']
    out = Path(settings['output'])
    out.mkdir(parents=True, exist_ok=True)
    runs = {}
    for jobs_dir in settings['jobs_dirs']:
        for key, run in load(Path(jobs_dir)).items():
            if key not in runs:
                runs[key] = {**run, 'sampling_verification': [run['sampling_verification']]}
                continue
            # The same model/setting/repeat served again for other panels (03-fixed-best): one run, two servers.
            if set(run['panels']) & set(runs[key]['panels']):
                raise ValueError(f'{key}: panels evaluated twice')
            runs[key]['panels'].update(run['panels'])
            runs[key]['scores'].update(run['scores'])
            runs[key]['sampling_verification'].append(run['sampling_verification'])
    models = list(dict.fromkeys(r['model'] for r in runs.values()))
    order = settings['setting_order']
    # Loop totals and paired counts use only the panels every run evaluated (fixed338 is default r1 only).
    core = set(settings['core_panels'])

    # Per-run, per-split metrics from AmphionEval's summary, plus the run total over the core panels.
    metrics = defaultdict(dict)
    for key, run in runs.items():
        rows = []
        for panel, values in run['panels'].items():
            for label, summary in values['summary']['labels'].items():
                metrics[key][f'{panel}/{label}'] = {f: summary[f] for f in FIELDS}
            if panel in core:
                rows += run['scores'][panel]
        metrics[key]['all/loops'] = sum(r['repetition_loop'] for r in rows)
        metrics[key]['all/samples'] = len(rows)

    # Common subset and paired loops per model.
    common, paired, loops_by_sample = {}, {}, {}
    for model in models:
        keys = [k for k in runs if runs[k]['model'] == model]
        looped = defaultdict(list)
        splits = defaultdict(dict)
        for key in keys:
            for panel, rows in runs[key]['scores'].items():
                for row in rows:
                    split = f"{panel}/{row['label']}"
                    splits[split].setdefault(key, {})[row['id']] = row
                    if row['repetition_loop']:
                        looped[(split, row['id'])].append(key)
        loops_by_sample[model] = {f'{s}/{i}': sorted(v) for (s, i), v in sorted(looped.items())}
        for split, by_run in splits.items():
            # Only runs that evaluated this split (fixed338 is in default r1 only).
            ids = set.intersection(*(set(rows) for rows in by_run.values()))
            clean = {i for i in ids if (split, i) not in looped}
            for key, rows in by_run.items():
                common.setdefault(key, {})[split] = {
                    'samples': len(clean), 'excluded': len(ids) - len(clean),
                    'cp_error_rate': rate([rows[i] for i in clean])}
        base = f'{model}/default/r1'
        for key in keys:
            if key == base or base not in runs:
                continue
            fixed = introduced = both = 0
            for (split, _), runs_looping in looped.items():
                if split.split('/')[0] not in core:
                    continue
                a, b = base in runs_looping, key in runs_looping
                fixed += a and not b
                introduced += b and not a
                both += a and b
            paired[key] = {'versus': base, 'fixed': fixed, 'introduced': introduced, 'both': both}

    # Rerun spread of default decoding.
    reruns = {}
    for model in models:
        keys = sorted(k for k in runs if runs[k]['model'] == model and runs[k]['setting'] == 'default')
        if len(keys) < 2:
            continue
        splits = set.intersection(*(set(m for m in metrics[k] if '/' in m and not m.startswith('all/'))
                                    for k in keys))
        row = {'runs': keys, 'all_loops': spread([metrics[k]['all/loops'] for k in keys]), 'splits': {}}
        for split in sorted(splits):
            row['splits'][split] = {
                'loops': spread([metrics[k][split]['repetition_loops'] for k in keys]),
                'cp_error_rate': spread([metrics[k][split]['cp_error_rate'] for k in keys]),
                'cp_error_rate_without_loops': spread([metrics[k][split]['cp_error_rate_without_loops'] for k in keys]),
                'common_cp_error_rate': spread([common[k][split]['cp_error_rate'] for k in keys])}
        flaky = defaultdict(int)
        for runs_looping in [v for v in loops_by_sample[model].values()]:
            hits = sum(k in runs_looping for k in keys)
            if hits:
                flaky[hits] += 1
        row['samples_looping_in_n_of_runs'] = dict(sorted(flaky.items()))
        reruns[model] = row

    verification = {k: r['sampling_verification'] for k, r in runs.items()}
    result = {'runs': {k: {'metrics': metrics[k], 'common': common[k]} for k in runs}, 'paired': paired,
              'reruns': reruns, 'loops_by_sample': loops_by_sample, 'sampling_verification': verification}
    (out / 'analysis.json').write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')

    # Markdown report.
    lines = ['# 解码设置与重跑波动汇总', '', '数值为百分比（cpER、DER、人数准确率）或条数（复读）。'
             '“共同子集”= 该模型所有执行中都未复读的样本，各设置/重复在同一批样本上比较。', '']
    rescored = [f'{k}/{p}' for k in sorted(runs) for p, v in runs[k]['panels'].items() if v.get('rescored')]
    lines += ['AmphionEval 读取问题后用 scripts/rescore.py 补分的评测：' + (', '.join(rescored) or '无'), '']
    splits = sorted({s for k in runs for s in metrics[k] if not s.startswith('all/')})
    for model in models:
        keys = sorted((k for k in runs if runs[k]['model'] == model),
                      key=lambda k: (order.index(runs[k]['setting']), runs[k]['repeat']))
        lines += [f'## {model}', '', '| 执行 | 核验 rp | 请求数 | 核心集复读（928 条） | 相对 default/r1 修复/新增/仍复读 |', '|---|---|---|---|---|']
        for k in keys:
            p = paired.get(k)
            pair = f"{p['fixed']}/{p['introduced']}/{p['both']}" if p else '—'
            v = verification[k]
            observed = sorted({x for check in v for x in check['observed']['repetition_penalty']})
            lines.append(f"| {k} | {observed} | {sum(c['requests'] for c in v)} | {metrics[k]['all/loops']} | {pair} |")
        lines += ['', '| 划分 | 执行 | 样本 | 复读 | cpER | 排除复读 cpER | 共同子集 cpER（样本数） | DER | 人数准确率 |',
                  '|---|---|---|---|---|---|---|---|---|']
        for split in splits:
            for k in keys:
                if split not in metrics[k]:
                    continue
                m, c = metrics[k][split], common[k][split]
                lines.append(f"| {split} | {k} | {m['samples']} | {m['repetition_loops']} | {pct(m['cp_error_rate'])} | "
                             f"{pct(m['cp_error_rate_without_loops'])} | {pct(c['cp_error_rate'])}（{c['samples']}） | "
                             f"{pct(m['der'])} | {pct(m['speaker_count_accuracy'])} |")
        lines += ['', '复读样本（样本 → 在哪些执行中复读）：', '']
        for sample, keys_looping in loops_by_sample[model].items():
            lines.append(f"- `{sample}`: {', '.join(keys_looping)}")
        lines.append('')
    if reruns:
        lines += ['## 默认解码重跑波动', '']
        for model, row in reruns.items():
            a = row['all_loops']
            lines += [f"### {model}（{', '.join(row['runs'])}）", '',
                      f"核心集复读条数：{a['values']}，范围 {a['min']}–{a['max']}，标准差 {a['std']:.2f}；"
                      f"在 n 次重跑中复读的样本数：{row['samples_looping_in_n_of_runs']}", '',
                      '| 划分 | 复读条数 | 复读标准差 | cpER 范围 | 排除复读 cpER 范围 | 极差 | 共同子集 cpER 范围 | 极差 |',
                      '|---|---|---|---|---|---|---|---|']
            for split, s in row['splits'].items():
                c, w, cc = s['cp_error_rate'], s['cp_error_rate_without_loops'], s['common_cp_error_rate']
                lines.append(f"| {split} | {s['loops']['values']} | {s['loops']['std']:.2f} | {pct(c['min'])}–{pct(c['max'])} | "
                             f"{pct(w['min'])}–{pct(w['max'])} | {100 * w['range']:.2f} | {pct(cc['min'])}–{pct(cc['max'])} | "
                             f"{100 * cc['range']:.2f} |")
            lines.append('')
    (out / 'analysis.md').write_text('\n'.join(lines) + '\n')
    # Flat numeric view for W&B.
    flat = {'runs': len(runs)}
    for k in runs:
        flat[f"{k.replace('/', '-')}/all_loops"] = metrics[k]['all/loops']
        for split, c in common[k].items():
            flat[f"{k.replace('/', '-')}/{split}/common_cp_error_rate"] = c['cp_error_rate']
    for k, p in paired.items():
        for f in ('fixed', 'introduced', 'both'):
            flat[f"{k.replace('/', '-')}/paired/{f}"] = p[f]
    for model, row in reruns.items():
        flat[f'{model}/rerun/all_loops_std'] = row['all_loops']['std']
        flat[f'{model}/rerun/all_loops_range'] = row['all_loops']['range']
        for split, s in row['splits'].items():
            flat[f'{model}/rerun/{split}/cp_error_rate_without_loops_range'] = s['cp_error_rate_without_loops']['range']
            flat[f'{model}/rerun/{split}/common_cp_error_rate_range'] = s['common_cp_error_rate']['range']
    (out / 'analysis-metrics.json').write_text(json.dumps(flat, indent=2) + '\n')


if __name__ == '__main__':
    main()
