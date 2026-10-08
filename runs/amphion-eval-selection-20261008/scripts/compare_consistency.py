"""Compare the served AmphionEval path with the experiment scripts on the same clips.

For each comparison three scores are reported per label, so a difference can be
assigned to its source:

- ``old``: the experiment script's own number (offline ``Qwen3ASRModel.LLM`` or
  the cached MOSS server output, scored by the experiment scorer);
- ``old_outputs_ae_scorer``: the same old outputs rescored by AmphionEval's
  meeting scorer (difference to ``old`` = scoring protocol);
- ``ae``: the new served outputs scored by AmphionEval (difference to the
  previous line = inference path).

Runs in the evaluation environment; it reads AmphionEval's frozen samples so the
references are exactly those the served evaluation used.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import yaml
from amphion_eval.integrations.open_audio_llm.meeting import TIME, aggregate, score_sample


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def old_scores(spec):
    summary = json.loads(Path(spec['file']).read_text())
    if spec['kind'] == 'selection':
        rows = summary['diarization']['by_candidate'][spec['candidate']]
        return {label: {'cp_error_rate': value['cpCER'], 'hit_token_limit': value['hit_token_limit_count']}
                for label, value in rows.items()}
    if spec['kind'] == 'ae':
        return summary['labels']
    rows = summary['candidates'][spec['candidate']]['by_dataset']
    return {label: {'cp_error_rate': value['cp_error_rate'], 'der': value['der'],
                    'repetition_loops': value['repetition_loops'], 'timed_coverage': value['timed_coverage'],
                    'speaker_count_accuracy': value['speaker_count_accuracy']} for label, value in rows.items()}


def macro(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def compare(spec):
    root = Path(spec['evaluation'])
    samples = {row['id']: row for row in read_jsonl(root / 'samples.jsonl')}
    new = {row['id']: row for row in read_jsonl(root / 'predictions.jsonl')}
    new_scores = {row['id']: row for row in read_jsonl(root / 'meeting-scores.jsonl')}
    old = {row['id']: row for row in read_jsonl(spec['old_predictions'])}
    assert set(old) == set(samples) == set(new), 'old and new evaluations must cover the same clips'
    field = 'raw_text' if spec['old_format'] == 'open_audio_llm' else 'text'
    output_format = 'open_audio_llm' if spec['old_format'] == 'open_audio_llm' else 'moss_transcribe_diarize'
    by_label = defaultdict(lambda: {'old': [], 'new': [], 'pairs': []})
    for sample_id, sample in samples.items():
        rescored = score_sample(sample['record'], old[sample_id][field], output_format, sample['duration'])
        current = new_scores[sample_id]
        rescore_new = score_sample(sample['record'], new[sample_id]['raw_text'], output_format, sample['duration'])
        assert rescore_new['cp_errors'] == current['cp_errors'], sample_id
        group = by_label[sample['label']]
        group['old'].append(rescored)
        group['new'].append(current)
        old_text = old[sample_id][field].split('<asr_text>', 1)[-1]
        new_text = new[sample_id]['raw_text'].split('<asr_text>', 1)[-1]
        prefix = next((i for i, (a, b) in enumerate(zip(old_text, new_text)) if a != b), min(len(old_text), len(new_text)))
        group['pairs'].append({'id': sample_id, 'identical_output': old_text == new_text,
                               'identical_without_timestamps': TIME.sub('', old_text) == TIME.sub('', new_text),
                               'common_prefix_fraction': prefix / max(len(old_text), 1),
                               'old_cp_errors': rescored['cp_errors'], 'new_cp_errors': current['cp_errors'],
                               'units': current['units'], 'old_loop': rescored['repetition_loop'],
                               'new_loop': current['repetition_loop']})
    reference = old_scores(spec['old_summary'])
    labels, details = {}, []
    for label, group in sorted(by_label.items()):
        pairs = group['pairs']
        deltas = [p['new_cp_errors'] - p['old_cp_errors'] for p in pairs]
        old_agg, new_agg = aggregate(group['old']), aggregate(group['new'])
        labels[label] = {
            'samples': len(pairs),
            'old': reference.get(label),
            'old_outputs_ae_scorer': old_agg,
            'ae': new_agg,
            'protocol_difference': old_agg['cp_error_rate'] - reference[label]['cp_error_rate'],
            'inference_difference': new_agg['cp_error_rate'] - old_agg['cp_error_rate'],
            'total_difference': new_agg['cp_error_rate'] - reference[label]['cp_error_rate'],
            'identical_outputs': sum(p['identical_output'] for p in pairs),
            'identical_without_timestamps': sum(p['identical_without_timestamps'] for p in pairs),
            'median_common_prefix_fraction': sorted(p['common_prefix_fraction'] for p in pairs)[len(pairs) // 2],
            'changed_cp_errors_samples': sum(d != 0 for d in deltas),
            'mean_abs_cp_error_delta_units': sum(map(abs, deltas)) / len(deltas),
            'loops_old_outputs': old_agg['repetition_loops'], 'loops_ae': new_agg['repetition_loops'],
            'loop_flips': sum(p['old_loop'] != p['new_loop'] for p in pairs),
            'largest_changes': sorted(pairs, key=lambda p: -abs(p['new_cp_errors'] - p['old_cp_errors']))[:5],
        }
        details += pairs
    result = {'name': spec['name'], 'evaluation': str(root), 'labels': labels}
    for group, members in (spec.get('groups') or {}).items():
        result.setdefault('groups', {})[group] = {
            key: macro([labels[m][key] if key != 'old' else labels[m]['old']['cp_error_rate'] for m in members])
            for key in ('old', 'protocol_difference', 'inference_difference', 'total_difference')}
        result['groups'][group].update(
            old_outputs_ae_scorer=macro([labels[m]['old_outputs_ae_scorer']['cp_error_rate'] for m in members]),
            ae=macro([labels[m]['ae']['cp_error_rate'] for m in members]),
            loops_old_outputs=sum(labels[m]['loops_old_outputs'] for m in members),
            loops_ae=sum(labels[m]['loops_ae'] for m in members))
    return result, details


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    settings = yaml.safe_load(parser.parse_args().config.read_text())['parameters']
    output = Path(settings['output'])
    output.mkdir(parents=True, exist_ok=True)
    report = {}
    for spec in settings['comparisons']:
        result, details = compare(spec)
        report[spec['name']] = result
        (output / f"{spec['name']}-samples.jsonl").write_text(
            ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in details))
    (output / 'consistency.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    # Flat headline numbers for W&B.
    metrics = {name: {**{f'group/{g}/{k}': v for g, vals in r.get('groups', {}).items() for k, v in vals.items()},
                      **{f'label/{label}/{k}': v for label, vals in r['labels'].items()
                         for k, v in vals.items() if isinstance(v, (int, float))}}
               for name, r in report.items()}
    (output / 'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
