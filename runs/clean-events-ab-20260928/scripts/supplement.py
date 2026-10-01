from settings import ROOT, RUNS, OUTPUT, SETTINGS, artifact, child
"""Fill English coverage using the repaired scheduler; preserve prior artifacts."""
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import gzip
import json
import multiprocessing
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import numpy as np

D = ROOT
sys.path.insert(0, str(artifact('synthesis-source-supplement')))
from amphiondata.multispeaker import events
import generate as generation
from prepare import save

def main():
    assert 'synthesis-source-supplement' in events.__file__
    started = time.time(); out = artifact('generated')
    recipe = json.loads((out / 'recipe.json').read_text())
    recipe['turn_policy'].update(recipe['language_policies']['en'])
    events.validate_recipe(recipe)
    events._ROOTS = json.loads((out / 'roots.json').read_text()); events._STOP_EVENT = None
    pool = defaultdict(lambda: defaultdict(list))
    for row in events.rows(artifact('word-pool.jsonl.gz')): pool[row['language']][row['speaker']].append(row)
    events._POOL = {l: dict(p) for l, p in pool.items()}
    from transformers import AutoTokenizer
    events._TOKENIZER = AutoTokenizer.from_pretrained(SETTINGS['parameters']['tokenizer'], local_files_only=True)
    generation.CENTROIDS = dict(zip(json.loads((artifact('centroid-speakers.json')).read_text()), np.load(artifact('centroids.npz'))['embeddings']))
    generation.RECIPES = {('en', False): recipe}
    tasks = []
    for seconds in (30., 120.):
        for scenario in events.SCENARIOS:
            cell = (scenario, 'en', 3, seconds)
            generation.ELIGIBLE[cell] = generation.natural_eligible(cell, recipe, events._POOL)
            for _ in range(10): tasks.append((1001 + len(tasks), cell, False))
    plan = {'tasks': len(tasks), 'language': 'en', 'durations': [30, 120],
        'reason': 'Observed long-fraction overflow: filler selection enforced the lower quota only.',
        'quality_thresholds_unchanged': True, 'generator_sha256': events.digest_file(Path(events.__file__)),
        'original_generator_sha256': events.digest_file(artifact('synthesis-source/amphiondata/multispeaker/events.py'))}
    save('supplement-plan.json', plan)
    evidence = artifact('supplement-results.jsonl')
    assert not evidence.exists(), 'Do not overwrite a previous supplement'
    good = []; bad = []
    with evidence.open('w') as stream, ProcessPoolExecutor(16, mp_context=multiprocessing.get_context('fork')) as ex:
        for f in as_completed([ex.submit(generation.job, t) for t in tasks]):
            result = f.result()
            if 'record' in result:
                result['record']['metadata']['generator_sha256'] = plan['generator_sha256']
                good.append(result)
            else: bad.append(result)
            stream.write(json.dumps(result, ensure_ascii=False) + '\n'); stream.flush()
            progress = {'completed': len(good) + len(bad), 'requested': len(tasks), 'generated': len(good),
                'quarantined': len(bad), 'hours': sum(x['index']['duration'] for x in good) / 3600,
                'elapsed_seconds': time.time() - started}
            save('generation-progress.json', {**progress, 'scope': 'English supplement only; original 457 records preserved'})
            print(json.dumps(progress), flush=True)
    folder = out / 'shards/00003'; folder.mkdir()
    counts = {k: Counter() for k in ('scenario', 'language', 'speakers', 'duration', 'cells', 'rejections')}
    with gzip.open(folder / 'records.jsonl.gz', 'wt') as a, gzip.open(folder / 'audio-index.jsonl.gz', 'wt') as b:
        for x in sorted(good, key=lambda x: x['number']):
            a.write(json.dumps(x['record'], ensure_ascii=False) + '\n'); b.write(json.dumps(x['index']) + '\n')
            sc, lang, n, sec = x['cell']
            for name, val in [('scenario', sc), ('language', lang), ('speakers', str(n)), ('duration', str(sec)), ('cells', '/'.join(map(str, x['cell'])))]: counts[name][val] += 1
            counts['rejections'].update(x['rejections'])
    shard = {'shard': 3, 'records': len(good), 'audio_seconds': sum(x['index']['duration'] for x in good),
        'counts': {k: dict(v) for k, v in counts.items()}, 'sha256': {p.name: events.digest_file(p) for p in [folder / 'records.jsonl.gz', folder / 'audio-index.jsonl.gz']}}
    events.write_json(folder / 'summary.json', shard)
    view = artifact('audit-3'); (view / 'shards').mkdir(parents=True)
    (view / 'shards/00000').symlink_to(folder, target_is_directory=True)
    (view / 'audio').symlink_to(out / 'audio', target_is_directory=True)
    events.write_json(view / 'recipe.json', recipe); events.audit(SimpleNamespace(output=view))
    summaries = [json.loads(p.read_text()) for p in sorted((out / 'shards').glob('*/summary.json'))]
    events._RECIPE = json.loads((out / 'recipe.json').read_text())
    events.publish(out, summaries, 'complete', 910 + len(tasks), started)
    records = [r for p in sorted((out / 'shards').glob('*/records.jsonl.gz')) for r in events.rows(p)]
    turns = [t for r in records for t in r['metadata']['turns']]
    responses = [t for t in turns if t['turn_kind'] == 'response']
    summary = {'records': len(records), 'hours': sum(r['audio_slots'][0]['ref']['duration'] for r in records) / 3600,
        'source_speakers': len({t['source_speaker'] for t in turns}), 'quarantined': 453 + len(bad),
        'language_counts': dict(Counter(r['language'] for r in records)),
        'duration_counts': dict(Counter(str(r['audio_slots'][0]['ref']['duration']) for r in records)),
        'scenario_counts': dict(Counter(r['labels']['scenario'] for r in records)),
        'response_turns': len(responses), 'unique_response_sources': len({t['source']['id'] for t in responses}),
        'response_scenes': sum(any(t['turn_kind'] == 'response' for t in r['metadata']['turns']) for r in records),
        'identity_threshold_calibrated': False, 'human_review_required': False, 'training_started': False}
    checks = {'records': len(records) >= 500, 'hours': summary['hours'] >= 10, 'source_speakers': summary['source_speakers'] >= 300,
        'duration_coverage': all(summary['duration_counts'].get(str(d), 0) >= 20 for d in (30., 120., 300.)),
        'scenario_coverage': all(summary['scenario_counts'].get(s, 0) >= 10 for s in events.SCENARIOS),
        'language_coverage': all(summary['language_counts'].get(l, 0) >= 100 for l in ('zh', 'en')),
        'real_response_coverage': summary['unique_response_sources'] >= 15 and summary['response_scenes'] >= 30}
    save('synthesis-summary.json', summary)
    save('release-gate.json', {'passed': all(checks.values()), 'checks': checks, 'summary': summary,
        'scope': 'Automatic-screened bounded experiment; thresholds unchanged.'})
    save('supplement-summary.json', {'generated': len(good), 'quarantined': len(bad), 'merged': summary, 'elapsed_seconds': time.time() - started})
    assert all(checks.values()), f'Release gate failed: {checks}'

if __name__ == '__main__': main()
