"""Regression for long-sentence-heavy English source reservoirs."""
import json
import argparse
from pathlib import Path
import random
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--experiment-root', type=Path, required=True)
args = parser.parse_args()
D = args.experiment_root.absolute()
sys.path.insert(0, str(D / 'shared/synthesis-source-supplement'))
from amphiondata.multispeaker import events

def test_long_heavy_pool_can_meet_upper_quota_without_cutting_words(seed):
    recipe = json.loads((D / 'tasks/03-synthesis/attempts/002/artifacts/generated/recipe.json').read_text())
    recipe['turn_policy']['annotation_overlap_ratio'] = [0., 1.]
    recipe['turn_policy']['response_fraction'] = [0., .15]
    pool = {'en': {}}
    for speaker in range(5):
        rows = []
        for i in range(100):
            duration = 9.8 if i < 75 else 2.8
            rows.append({'id': f'{speaker}-{i}', 'language': 'en', 'text': 'This is a complete sentence.',
                'words': [{'start': .1, 'end': .1 + duration}], 'speaker': str(speaker)})
        pool['en'][str(speaker)] = rows
    turns, measured, _, _, _ = events.make_turns(pool, recipe, random.Random(seed), 'en', 3, 120., 'overlap_entry')
    assert .05 <= measured['long_fraction'] <= .4
    assert all(t['text'] == t['source']['text'] and t['word_range'] == [0, 1] for t in turns)

for seed in (7, 19, 42):
    test_long_heavy_pool_can_meet_upper_quota_without_cutting_words(seed)
print('3 scheduler checks passed')
