"""Score a finished AmphionEval meeting run whose JSONL reader split a model output on U+2028.

AmphionEval 0.7.0 reads worker-predictions.jsonl with ``str.splitlines()``, which also breaks on
Unicode line separators (U+2028, U+0085, ...) that ``json.dumps(ensure_ascii=False)`` writes raw.
Under repetition penalty the model sometimes emits U+2028 instead of a newline; the reader then
fails that sample and every later one although the server answered all of them.

This script re-reads the same files splitting only on ``\n`` and scores with AmphionEval 0.7.0's own
``score_sample`` and ``summarize`` in the same sample order, so normalization and scoring are those
of the original run. It refuses unless every sample has exactly one successful server answer.
Run it with the amphion-eval interpreter.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path

from amphion_eval.integrations.open_audio_llm.meeting import score_sample, summarize


def jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').split('\n') if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('evaluation', type=Path)
    parser.add_argument('--output', type=Path, help='defaults to the evaluation directory')
    options = parser.parse_args()
    root, out = options.evaluation, options.output or options.evaluation
    settings = json.loads((root / 'settings.json').read_text())
    samples = jsonl(root / 'samples.jsonl')
    answers = {}
    for row in jsonl(root / 'worker-predictions.jsonl'):
        if row['id'] in answers:
            raise ValueError(f"duplicate answer {row['id']}")
        answers[row['id']] = row
    missing = [s['id'] for s in samples if answers.get(s['id'], {}).get('status') != 'success']
    if missing or len(answers) != len(samples):
        raise ValueError(f'{len(missing)} samples lack a successful answer; not a reader-only failure')
    scored = defaultdict(list)
    for sample in samples:
        raw = answers[sample['id']]['raw_text']
        scored[sample['label']].append({
            **score_sample(sample['record'], raw, settings['effective']['output_format'], sample['duration']),
            'label': sample['label']})
    rows = [item for items in scored.values() for item in items]
    (out / 'meeting-scores-rescored.jsonl').write_text(
        ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')
    summary = summarize(scored, settings['data'].get('groups'))
    separators = sum(any(c in answers[s['id']]['raw_text'] for c in '\u2028\u2029\x85\x1c\x1d\x1e\x0b\x0c')
                     for s in samples)
    summary['rescore'] = {'reason': 'AmphionEval 0.7.0 splitlines() on worker-predictions.jsonl',
                          'samples': len(samples), 'outputs_with_unicode_line_separators': separators}
    (out / 'summary-rescored.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()
