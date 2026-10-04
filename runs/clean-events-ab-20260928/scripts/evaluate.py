"""Fixed 338-clip comparison with vLLM; W&B is owned by the shared launcher."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from settings import ROOT, SETTINGS, artifact, child

E = artifact('evaluation')
BASE = Path(SETTINGS['parameters']['reference_experiment'])
PREVIOUS = Path(SETTINGS['parameters']['reference_evaluation'])


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    resuming = SETTINGS['recording'].get('resuming', False)
    E.mkdir(parents=True, exist_ok=resuming)
    plan = json.loads((PREVIOUS / 'plan.json').read_text())
    plan.update(worker_count=1, models={}, comparison='treatment minus matched control at 1000 steps',
                training_steps_per_arm=1000, evaluate_initial_checkpoint=True,
                primary_metric='tcpCER_5s', supporting_metric='cpCER', optional_stopping=False,
                scope='Joint source-quality and synthesis-pipeline effect; not separate causal attribution.')
    model_root = E / 'models'
    model_root.mkdir(exist_ok=resuming)
    for candidate in ('control', 'treatment', 'initial'):
        source = Path(SETTINGS['parameters']['models'][candidate])
        dest = model_root / candidate
        if not dest.exists():
            dest.mkdir()
            for item in source.iterdir():
                if item.name in {'config.json', 'args.json', 'trainer_state.json', 'catalog_sampler.json'}:
                    continue
                if item.suffix in {'.json', '.jinja', '.safetensors', '.txt'}:
                    (dest / item.name).symlink_to(item.resolve())
            cfg = json.loads((source / 'config.json').read_text())
            assert cfg['thinker_config']['audio_config']['n_window_infer'] == 800
            cfg['thinker_config']['audio_config']['n_window_infer'] = SETTINGS['parameters']['encoder_window']
            (dest / 'config.json').write_text(json.dumps(cfg, indent=2) + '\n')
        plan['models'][candidate] = str(dest)
    clips_path = E / 'clips.jsonl'
    if not clips_path.exists():
        shutil.copy2(PREVIOUS / 'clips.jsonl', clips_path)
    assert sha(clips_path) == plan['manifest_sha256']
    clips = list(map(json.loads, clips_path.read_text().splitlines()))
    assert len(clips) == 338
    for clip in clips:
        assert sha(clip['audio']) == clip['audio_sha256']
    (E / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    children = []
    try:
        for batch in [('control', 'treatment'), ('initial',)]:
            jobs = {}
            for position, candidate in enumerate(batch):
                folder = E / candidate
                folder.mkdir(exist_ok=resuming)
                complete = folder / 'shard-0/inference-complete.json'
                if complete.exists():
                    continue
                # A shard can resume only from complete append-only records.
                with (folder / 'inference.log').open('a') as log:
                    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(SETTINGS['runtime']['gpus'][position])}
                    process = subprocess.Popen(
                        child('infer.py', candidate, '--shard', 0,
                              python=SETTINGS['parameters']['inference_python']),
                        env=env, stdout=log, stderr=subprocess.STDOUT)
                jobs[candidate] = process
                children.append(process)
            for candidate, process in jobs.items():
                code = process.wait()
                if code:
                    raise RuntimeError(f'{candidate} inference exited {code}')
            for candidate in batch:
                folder = E / candidate
                rows = list(map(json.loads, (folder / 'shard-0/predictions.jsonl').read_text().splitlines()))
                assert len(rows) == 338 and {r['id'] for r in rows} == {c['id'] for c in clips}
                runtime = json.loads((folder / 'shard-0/runtime.json').read_text())
                assert runtime['backend'] == 'vllm'
                shutil.copy2(folder / 'shard-0/predictions.jsonl', folder / 'predictions.jsonl')
        reference = json.loads((E / 'control/shard-0/runtime.json').read_text())
        for name in ('treatment', 'initial'):
            runtime = json.loads((E / name / 'shard-0/runtime.json').read_text())
            assert {k: v for k, v in runtime.items() if k != 'candidate'} == {
                k: v for k, v in reference.items() if k != 'candidate'}
        subprocess.run(child('score.py', python=SETTINGS['parameters']['scoring_python']),
                       env={**os.environ, 'CUDA_VISIBLE_DEVICES': ''}, check=True)
        summary = json.loads((E / 'summary.json').read_text())
        assert summary['backend'] == 'vllm'
    finally:
        for process in children:
            if process.poll() is None:
                process.terminate()
                process.wait()


if __name__ == '__main__':
    main()
