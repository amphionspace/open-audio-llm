"""Run AmphionEval after-training for MOSS-Transcribe-Diarize as a TS-ASR JSONL worker.

All settings come from the task YAML: this writes the handoff and the after-training
config into the attempt and runs ``ae open-audio-llm after-training`` with them, so
data, normalization, scoring and W&B logging are those of the AmphionASR TS runs.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    settings = yaml.safe_load(parser.parse_args().config.read_text())['parameters']
    output = Path(settings['output'])
    output.mkdir(parents=True, exist_ok=True)
    worker = settings['worker']
    handoff = output / 'handoff.json'
    handoff.write_text(json.dumps({
        'framework': 'open-audio-llm', 'schema_version': 2, 'checkpoint': settings['model'], 'base_model': '',
        'tuner_type': 'full', 'selection': 'final', 'global_step': 0, 'training_run': settings['model']}) + '\n')
    command = [worker['python'], worker['script'], '--input', '{input}', '--output', '{output}', '--model', '{model}']
    for key, value in worker['arguments'].items():
        command += [f'--{key}', str(value)]
    config = {
        'suite_id': settings['suite_id'], 'model_name': settings['model_name'], 'evaluator': 'ts-asr',
        'eval_set': settings['eval_set'], 'worker_command': command, 'worker_tasks': ['ts_asr'],
        'worker_timeout': settings['worker_timeout'], 'output_root': str(output / 'ae'),
        'parameters': settings['parameters'], 'wandb': settings['wandb'], 'publish': settings['publish'],
    }
    path = output / 'after-training.yaml'
    path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False))
    sys.exit(subprocess.run([settings['ae'], 'open-audio-llm', 'after-training', '--handoff', str(handoff),
                             '--config', str(path)]).returncode)


if __name__ == '__main__':
    main()
