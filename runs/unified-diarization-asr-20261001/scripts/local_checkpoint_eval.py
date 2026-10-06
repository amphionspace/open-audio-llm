"""Merge a saved adapter on CPU, then run tracked fixed-set vLLM evaluation."""
import json
import os
from pathlib import Path
import subprocess
import sys

import yaml


def main(checkpoint, output):
    checkpoint, output = Path(checkpoint).resolve(), Path(output).resolve()
    training = yaml.safe_load((checkpoint.parents[2] / 'effective.yaml').read_text())
    settings = training['parameters']['checkpoint_evaluation']
    step = int(checkpoint.name.removeprefix('checkpoint-'))
    merged = output / 'merged-model'
    merge_code = '''
import sys, shutil
from pathlib import Path
import torch
from qwen_asr import Qwen3ASRModel
from peft import PeftModel
checkpoint, output, base = map(Path, sys.argv[1:])
model = Qwen3ASRModel.from_pretrained(str(base), dtype=torch.bfloat16, device_map='cpu').model
model = PeftModel.from_pretrained(model, str(checkpoint)).merge_and_unload(safe_merge=True)
model.save_pretrained(output, safe_serialization=True)
for source in base.iterdir():
    if source.suffix in {'.json', '.txt', '.jinja'} and source.name not in {
        'config.json', 'args.json', 'trainer_state.json', 'catalog_sampler.json',
        'model.safetensors.index.json'} and not (output / source.name).exists():
        shutil.copy2(source, output / source.name)
'''
    subprocess.run([sys.executable, '-c', merge_code, str(checkpoint), str(merged),
                    training['task']['arguments']['--model']],
                   env={**os.environ, 'CUDA_VISIBLE_DEVICES': ''}, check=True)
    # load_config resolves paths against the original template before copying it.
    from open_audio_llm.run_config import load_config
    config = load_config(Path(settings['template']), output / 'eval-attempt')
    config['runtime']['gpus'] = [0]
    config['parameters'].update(
        output={'path': '{attempt}/artifacts/evaluation'},
        primary_candidate='latest', training_step=step,
        models={'latest': str(merged)},
        gpu_memory_utilization=settings['gpu_memory_utilization'],
        cached_predictions={'unified': settings['baseline_predictions']},
        cached_asr_predictions={'unified': settings['baseline_asr_predictions']},
    )
    config['recording'] = {'attempts_dir': {'path': str(output / 'attempts')},
                           'result': {'path': '{attempt}/artifacts/evaluation/summary.json'}}
    config['tracking'].update(name=f"{settings['run_name']}-step-{step}")
    config['tracking']['metrics'] = [{'file': {'path': '{attempt}/artifacts/evaluation/summary.json'},
                                       'prefix': 'evaluation'}]
    config['task']['script'] = {'path': config['task']['script']}
    # Resolved ordinary path values must retain path markers for formal loading.
    for key in ('python', 'cwd'):
        config['runtime'][key] = {'path': config['runtime'][key]}
    config['runtime']['pythonpath'] = [{'path': p} for p in config['runtime']['pythonpath']]
    for key in ('python', 'credentials_file'):
        config['tracking'][key] = {'path': config['tracking'][key]}
    config['tracking']['pythonpath'] = [{'path': p} for p in config['tracking']['pythonpath']]
    recipe = output / 'evaluation.yaml'
    recipe.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True))
    subprocess.run([sys.executable, '-m', 'open_audio_llm.cli', 'eval', '--config', str(recipe)], check=True)
    attempt = max((output / 'attempts').glob('[0-9]*'))
    summary = json.loads((attempt / 'artifacts/evaluation/summary.json').read_text())
    metrics = {}
    for dataset, scores in summary['diarization']['by_candidate']['latest'].items():
        for key in ('cpCER', 'tcpCER_5s', 'tcpCER_1s', 'hit_token_limit_count'):
            if isinstance(scores.get(key), (int, float)):
                metrics[f'eval/{dataset}/{key}'] = scores[key]
    for source, scores in summary['asr_regression']['latest'].items():
        metrics[f'eval/asr/{source}/error_rate'] = scores['error_rate']
    (output / 'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')


if __name__ == '__main__':
    main(*sys.argv[1:])
