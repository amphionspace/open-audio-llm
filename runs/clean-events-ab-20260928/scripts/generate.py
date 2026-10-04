from settings import ROOT, RUNS, OUTPUT, SETTINGS, artifact, child
"""Generate an automatically screened pilot with explicit language policies."""
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import gzip
import hashlib
import itertools
import json
import multiprocessing
from pathlib import Path
import random
import sys
import time
from types import SimpleNamespace
import numpy as np
from prepare import save

D = ROOT
sys.path.insert(0, str(artifact('synthesis-source')))
from amphiondata.multispeaker import events
OUT = artifact('generated')
CENTROIDS = {}
ELIGIBLE = {}
RECIPES = {}

def recipe():
    old = RUNS / 'identity-synthesis-20260927'
    r = json.loads((old / 'recipe-pilot.json').read_text())
    r.update(dataset_id='sot_speaker_events_clean_ab', version='events-auto-ab-20260928', seed=20260928,
        duration_weights={'30': 20, '120': 50, '300': 30}, speaker_group_weights={'2-4': 80, '5-6': 20}, max_sample_attempts=200)
    r['turn_policy'] = json.loads((artifact('source-turn-policy.json')).read_text())
    r['turn_policy']['response_fraction'] = [0., .35]
    r['turn_policy']['long_fraction'] = [.05, .4]
    r['acoustics'] = json.loads((old / 'repair-005-light-room/repair.json').read_text())['recipe']['acoustics']
    room = Path(r['acoustics']['rirs'][0]['path']).parent
    r['acoustics']['rirs'] = [{'path': str(p), 'sha256': events.digest_file(p)} for p in sorted(room.glob('*.wav'))[:6]]
    r['language_policies'] = {'zh': {'response_fraction': [0., .35]}, 'en': {'response_fraction': [0., .15]}}
    r['focused_response_policy'] = {'language': 'zh', 'response_fraction': [.10, .35],
        'scope': 'Extra short sessions using actual recorded acknowledgments; separately counted.'}
    return r

def natural_eligible(cell, r, pool):
    sc, lang, count, seconds = cell
    windows, anchors, _ = events.scenario_windows(sc, count, seconds)
    policy = r['turn_policy']
    windows = [[(a, min(b, seconds - policy['tail_seconds'])) for a, b in blocks] for blocks in windows]
    speech = {s: sum(x['words'][-1]['end'] - x['words'][0]['start'] for x in rs) for s, rs in pool[lang].items()}
    candidates = events.eligible_roles(speech, windows, r['source_speech_reserve_ratio'])
    budgets = defaultdict(list)
    for role, onset, _ in anchors:
        stop = next(b for a, b in windows[role] if a <= onset < b)
        stop = min([stop] + [t for s, t, _ in anchors if s == role and t > onset])
        budgets[role].append(min(r['turn_seconds'][1], stop - onset))
    result = []
    for role, names in enumerate(candidates):
        bounds = sorted(budgets[role]); kept = []
        for name in names:
            lengths = sorted(x['words'][-1]['end'] - x['words'][0]['start'] for x in pool[lang][name])
            lengths = [x for x in lengths if x >= policy['minimum_seconds']]
            if len(lengths) >= len(bounds) and all(x <= b for x, b in zip(lengths, bounds)): kept.append(name)
        result.append(kept)
    return result

def job(task):
    number, cell, focused = task
    recipe_key = (cell[1], focused)
    events._RECIPE = RECIPES[recipe_key]
    rng = random.Random(20260928 + number)
    # A different deterministic candidate roster per scene bounds scheduling cost
    # while retaining the expanded pool across scenes.
    events._ELIGIBLE = {cell: [rng.sample(names, min(24, len(names))) for names in ELIGIBLE[cell]]}
    try:
        record, index, rejected = events.generate_record(number, cell, OUT)
        voices = sorted({t['source_speaker'] for t in record['metadata']['turns']})
        for a, b in itertools.combinations(voices, 2):
            same_native = (a.split(':')[0], a.rsplit(':', 1)[-1]) == (b.split(':')[0], b.rsplit(':', 1)[-1])
            if same_native or float(CENTROIDS[a] @ CENTROIDS[b]) >= .65:
                return {'number': number, 'cell': cell, 'error': 'cross_role_identity_uncertain', 'focused': focused}
        record['metadata']['turn_policy'] = events._RECIPE['turn_policy']
        record['metadata']['focused_response_sample'] = focused
        record['metadata']['identity_screen'] = {'pass': True, 'cross_role_max_cosine': max(float(CENTROIDS[a] @ CENTROIDS[b]) for a, b in itertools.combinations(voices, 2)), 'threshold': .65, 'threshold_calibrated': False}
        # This is an actual derived-mixture screen, not a copied source flag.
        record['metadata']['clean'] = {'pass': True, 'scope': 'automatic_pipeline_checks',
            'checks': ['actual_source_clean_pass', 'source_identity_cohort', 'whole_source_utterances', 'event_timeline', 'acoustic_masking', 'cross_role_identity'],
            'human_verified': False, 'identity_threshold_calibrated': False}
        return {'number': number, 'cell': cell, 'focused': focused, 'record': record, 'index': index, 'rejections': dict(rejected)}
    except Exception as exc:
        return {'number': number, 'cell': cell, 'focused': focused, 'error': f'{type(exc).__name__}: {exc}'}

def main():
    global CENTROIDS, ELIGIBLE, RECIPES
    started = time.time(); r = recipe(); events.validate_recipe(r)
    OUT.mkdir(exist_ok=False)
    roots = json.loads((artifact('roots.json')).read_text()); roots[events.ROOT_ALIAS] = str(OUT)
    events._ROOTS = roots; events._STOP_EVENT = None
    pool = defaultdict(lambda: defaultdict(list))
    for row in events.rows(artifact('word-pool.jsonl.gz')): pool[row['language']][row['speaker']].append(row)
    events._POOL = {l: dict(p) for l, p in pool.items()}
    from transformers import AutoTokenizer
    events._TOKENIZER = AutoTokenizer.from_pretrained(SETTINGS['parameters']['tokenizer'], local_files_only=True)
    speakers = json.loads((artifact('centroid-speakers.json')).read_text())
    matrix = np.load(artifact('centroids.npz'))['embeddings']; CENTROIDS = dict(zip(speakers, matrix))
    for lang in ('zh', 'en'):
        rec = copy.deepcopy(r); rec['turn_policy'].update(r['language_policies'][lang]); RECIPES[lang, False] = rec
    focused = copy.deepcopy(RECIPES['zh', False]); focused['turn_policy']['response_fraction'] = [.10, .35]
    RECIPES['zh', True] = focused
    tasks = []
    # 7 scenarios x (zh:6 + en:4) repeats x 3 lengths x 2 speaker counts = 420.
    # Two independent blocks request 840 general scenes, plus 70 reply scenes;
    # failures are quarantined and never replaced with relaxed quality checks.
    for block in range(2):
        for sc in events.SCENARIOS:
            for lang, repeats in [('zh', 6), ('en', 4)]:
                for seconds, counts in [(30., [3, 4]), (120., [3, 4]), (300., [4, 6])]:
                    for count in counts:
                        cell = (sc, lang, count, seconds)
                        if cell not in ELIGIBLE: ELIGIBLE[cell] = natural_eligible(cell, r, events._POOL)
                        if any(not names for names in ELIGIBLE[cell]): continue
                        for _ in range(repeats): tasks.append((len(tasks) + 1, cell, False))
    for sc in events.SCENARIOS:
        cell = (sc, 'zh', 3, 30.)
        if any(not names for names in ELIGIBLE.get(cell, [])): continue
        for _ in range(10): tasks.append((len(tasks) + 1, cell, True))
    events.write_json(OUT / 'recipe.json', r); events.write_json(OUT / 'roots.json', roots)
    events.write_json(OUT / 'sources.json', {'pool': str(artifact('word-pool.jsonl.gz')), 'pool_sha256': events.digest_file(artifact('word-pool.jsonl.gz')),
         'generator_sha256': events.digest_file(Path(events.__file__)), 'identity_threshold_calibrated': False,
         'source_summary': json.loads((artifact('clean-pool-summary.json')).read_text())})
    save('generation-plan.json', {'tasks': len(tasks), 'minimum_records': 500, 'minimum_hours': 10, 'minimum_source_speakers': 300,
        'quality_thresholds_unchanged': True, 'cells': [list(t[1]) + [t[2]] for t in tasks]})
    good = []; bad = []; completed = 0
    with (artifact('generation-results.jsonl')).open('w') as evidence:
        with ProcessPoolExecutor(12, mp_context=multiprocessing.get_context('fork')) as executor:
            fs = [executor.submit(job, t) for t in tasks]
            for future in as_completed(fs):
                result = future.result(); completed += 1
                (good if 'record' in result else bad).append(result)
                evidence.write(json.dumps(result, ensure_ascii=False) + '\n'); evidence.flush()
                progress = {'completed': completed, 'requested': len(tasks), 'generated': len(good), 'quarantined': len(bad),
                    'hours': sum(x['index']['duration'] for x in good) / 3600, 'elapsed_seconds': time.time() - started}
                save('generation-progress.json', progress)
                print(json.dumps(progress), flush=True)
    # Each shard is audited against its actual language/response policy.
    summaries = []
    for shard, key in enumerate([('zh', False), ('en', False), ('zh', True)]):
        group = sorted([x for x in good if (x['cell'][1], x['focused']) == key], key=lambda x: x['number'])
        if not group: continue
        folder = OUT / 'shards' / f'{shard:05d}'; folder.mkdir(parents=True)
        counts = {k: Counter() for k in ('scenario', 'language', 'speakers', 'duration', 'cells', 'rejections')}
        with gzip.open(folder / 'records.jsonl.gz', 'wt') as a, gzip.open(folder / 'audio-index.jsonl.gz', 'wt') as b:
            for x in group:
                a.write(json.dumps(x['record'], ensure_ascii=False) + '\n'); b.write(json.dumps(x['index']) + '\n')
                sc, lang, n, sec = x['cell']
                for name, val in [('scenario', sc), ('language', lang), ('speakers', str(n)), ('duration', str(sec)), ('cells', '/'.join(map(str, x['cell'])))]: counts[name][val] += 1
                counts['rejections'].update(x['rejections'])
        summary = {'shard': shard, 'records': len(group), 'audio_seconds': sum(x['index']['duration'] for x in group),
            'counts': {k: dict(v) for k, v in counts.items()}, 'sha256': {p.name: events.digest_file(p) for p in [folder / 'records.jsonl.gz', folder / 'audio-index.jsonl.gz']}}
        events.write_json(folder / 'summary.json', summary); summaries.append(summary)
    # Existing audit supports one policy per invocation. Give each shard a
    # temporary audit view, without altering records or source audio.
    for shard, summary in enumerate(summaries):
        actual_shard = summary['shard']; key = [('zh', False), ('en', False), ('zh', True)][actual_shard]
        view = artifact(f'audit-{actual_shard}'); (view / 'shards').mkdir(parents=True)
        (view / 'shards/00000').symlink_to(OUT / 'shards' / f'{actual_shard:05d}', target_is_directory=True)
        (view / 'audio').symlink_to(OUT / 'audio', target_is_directory=True)
        events.write_json(view / 'recipe.json', RECIPES[key]); events.audit(SimpleNamespace(output=view))
    events._RECIPE = r
    events.publish(OUT, summaries, 'complete', len(tasks), started)
    seconds = sum(x['index']['duration'] for x in good)
    used = {t['source_speaker'] for x in good for t in x['record']['metadata']['turns']}
    response_turns = [t for x in good for t in x['record']['metadata']['turns'] if t['turn_kind'] == 'response']
    summary = {'records': len(good), 'hours': seconds / 3600, 'source_speakers': len(used), 'quarantined': len(bad),
        'language_counts': dict(Counter(x['cell'][1] for x in good)), 'duration_counts': dict(Counter(str(x['cell'][3]) for x in good)),
        'scenario_counts': dict(Counter(x['cell'][0] for x in good)), 'response_turns': len(response_turns),
        'unique_response_sources': len({t['source']['id'] for t in response_turns}),
        'response_scenes': sum(any(t['turn_kind'] == 'response' for t in x['record']['metadata']['turns']) for x in good),
        'identity_threshold_calibrated': False, 'human_review_required': False, 'training_started': False}
    save('synthesis-summary.json', summary); save('generation-failures.json', bad)
    checks = {'records': len(good) >= 500, 'hours': seconds >= 36000, 'source_speakers': len(used) >= 300,
        'duration_coverage': all(summary['duration_counts'].get(str(d), 0) >= 20 for d in (30., 120., 300.)),
        'scenario_coverage': all(summary['scenario_counts'].get(s, 0) >= 10 for s in events.SCENARIOS),
        'language_coverage': all(summary['language_counts'].get(l, 0) >= 100 for l in ('zh', 'en')),
        'real_response_coverage': summary['unique_response_sources'] >= 15 and summary['response_scenes'] >= 30}
    save('release-gate.json', {'passed': all(checks.values()), 'checks': checks, 'summary': summary,
        'scope': 'Automatic-screened data for a bounded 1000-step experiment, not a human-clean gold release.'})
    assert all(checks.values()), f'Release gate failed: {checks}'
    print(json.dumps(summary), flush=True)

if __name__ == '__main__': main()
