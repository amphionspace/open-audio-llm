"""Score every development mixture; preserve failures and repetition in cpER."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import yaml
from score_benchmark import aggregate, score_sample


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    settings = yaml.safe_load(parser.parse_args().config.read_text())['parameters']
    output = Path(settings['output'])
    clips = [json.loads(x) for x in (output/'clips.jsonl').read_text().splitlines()]
    candidate = settings['primary_candidate']
    predictions = {r['id']:r for r in map(json.loads,(output/candidate/'predictions.jsonl').read_text().splitlines())}
    assert predictions.keys() == {c['id'] for c in clips}
    groups, scored = defaultdict(list), []
    for clip in clips:
        row = score_sample(clip,predictions[clip['id']],'open_audio_llm')
        groups[str(clip['reference_speakers'])].append(row)
        scored.append(row)
    by_speakers = {n:aggregate(rs) for n,rs in groups.items()}
    summary = {'candidate':candidate,'samples':len(clips),'backend':'vllm',
               'by_speakers':by_speakers,'all':aggregate(scored),
               'macro_cpER':sum(r['cp_error_rate'] for r in by_speakers.values())/len(by_speakers),
               'protocol':'AmphionEval 0.6.0 diarization; all samples included'}
    (output/'selection-scores.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in scored))
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')


if __name__ == '__main__':
    main()
