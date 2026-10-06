"""Evaluate fixed meeting clips and ASR regression before continuous training."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    settings = config['parameters']
    output = Path(settings['output'])
    output.mkdir(parents=True, exist_ok=True)
    plan = json.loads(Path(settings['reference_plan']).read_text())
    plan.update(worker_count=1, models={})
    manifest = output / 'clips.jsonl'
    shutil.copy2(settings['reference_clips'], manifest)
    assert hashlib.sha256(manifest.read_bytes()).hexdigest() == plan['manifest_sha256']
    for candidate, model in settings['models'].items():
        source = Path(model)
        view = output / 'models' / candidate
        view.mkdir(parents=True, exist_ok=True)
        for item in source.iterdir():
            if item.name in {'config.json', 'args.json', 'trainer_state.json', 'catalog_sampler.json'}:
                continue
            if item.suffix in {'.json', '.jinja', '.safetensors', '.txt'}:
                (view / item.name).symlink_to(item.resolve())
        model_config = json.loads((source / 'config.json').read_text())
        model_config['thinker_config']['audio_config']['n_window_infer'] = plan['encoder_attention']['n_window_infer']
        (view / 'config.json').write_text(json.dumps(model_config, indent=2) + '\n')
        plan['models'][candidate] = str(view)
    (output / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    jobs = []
    try:
        for gpu, candidate in zip(config['runtime']['gpus'], settings['models']):
            folder = output / candidate
            folder.mkdir(exist_ok=True)
            if settings.get('completed_inference'):
                cached = Path(settings['completed_inference']) / candidate
                completion = json.loads((cached / 'inference-complete.json').read_text())
                assert completion == {'asr_rows': 96, 'backend': 'vllm'}
                for name in ('predictions.jsonl', 'asr-predictions.jsonl', 'runtime.json',
                             'inference-complete.json'):
                    shutil.copy2(cached / name, folder / name)
                continue
            with (folder / 'inference.log').open('a') as log:
                process = subprocess.Popen(
                    [sys.executable, str(Path(__file__).with_name('infer.py')),
                     '--config', str(args.config), '--candidate', candidate],
                    env={**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu)},
                    stdout=log, stderr=subprocess.STDOUT,
                )
            jobs.append(process)
        for process in jobs:
            code = process.wait()
            if code:
                raise RuntimeError(f'vLLM inference exited {code}')
        subprocess.run(
            [settings['scoring_python'], str(Path(__file__).with_name('score.py')),
             '--config', str(args.config)],
            env={**os.environ, 'CUDA_VISIBLE_DEVICES': ''}, check=True,
        )
    finally:
        for process in jobs:
            if process.poll() is None:
                process.terminate()
                process.wait()


if __name__ == '__main__':
    main()
