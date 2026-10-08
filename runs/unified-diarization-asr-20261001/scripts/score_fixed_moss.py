"""Score MOSS on the fixed 338-clip set with the archived MOSS-comparison scorer.

Compares the official-vLLM MOSS outputs with the earlier native FP32 MOSS outputs and
our checkpoints on identical clips, so the MOSS target can be restated on the
inference path the project rules prefer (official vLLM when supported).
"""
import argparse
import json
import runpy
from collections import defaultdict
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    settings = yaml.safe_load(parser.parse_args().config.read_text())['parameters']
    helpers = runpy.run_path(settings['scoring_helpers'])
    clips = [json.loads(line) for line in Path(settings['clips']).read_text().splitlines()]
    output = Path(settings['output'])
    output.mkdir(parents=True, exist_ok=True)
    scored = {}
    for name, spec in settings['candidates'].items():
        rows = {r['id']: r for r in map(json.loads, Path(spec['file']).read_text().splitlines())}
        assert set(rows) == {c['id'] for c in clips}, name
        result = []
        for clip in clips:
            row = {'hit_token_limit': False, 'generated_tokens': None, 'inference_seconds': None,
                   **rows[clip['id']]}
            result.append(helpers['score_one'](clip, row, spec['format']))
        scored[name] = result
    summary = {'candidates': {}, 'comparisons': {}}
    for name, rows in scored.items():
        timed = {r['id'] for r in rows if r['timed_available']}
        by = {ds: helpers['aggregate']([r for r in rows if r['dataset'] == ds], timed)
              for ds in ('aishell4', 'alimeeting')}
        summary['candidates'][name] = {
            'cpCER_macro': sum(by[d]['cpCER'] for d in by) / 2,
            'by_dataset': {d: {k: by[d].get(k) for k in ('cpCER', 'tcpCER_5s', 'timed_coverage',
                                                          'speaker_count_accuracy', 'hit_token_limit_count')}
                           for d in by}}
    primary = settings['primary']
    for name, rows in scored.items():
        if name == primary:
            continue
        groups = {ds: defaultdict(lambda: [0, 0, 0]) for ds in ('aishell4', 'alimeeting')}
        for a, b in zip(scored[primary], rows):
            totals = groups[a['dataset']][a['meeting_id']]
            totals[0] += a['cp_errors']
            totals[1] += b['cp_errors']
            totals[2] += a['reference_characters']
        stats = helpers['uncertainty']({ds: list(g.values()) for ds, g in groups.items()})
        summary['comparisons'][f'{primary}_vs_{name}'] = stats['equal_weight_corpus_macro']
    (output / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({n: round(c['cpCER_macro'], 4) for n, c in summary['candidates'].items()}))


if __name__ == '__main__':
    main()
